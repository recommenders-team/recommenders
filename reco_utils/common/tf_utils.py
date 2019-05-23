# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools

import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_DIR = "model_checkpoints"


def pandas_input_fn(
    df, y_col=None, batch_size=128, num_epochs=1, shuffle=False, seed=None
):
    """Pandas input function for TensorFlow high-level API Estimator.
    This function returns tf.data.Dataset function.

    Note. tf.estimator.inputs.pandas_input_fn cannot handle array/list column properly.
    For more information, see (https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn)

    Args:
        df (pd.DataFrame): Data containing features.
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function.
        num_epochs (int): Number of epochs to iterate over data. If None will run forever.
        shuffle (bool): If True, shuffles the data queue.
        seed (int): Random seed for shuffle.

    Returns:
        tf.data.Dataset function
    """

    X_df = df.copy()
    if y_col is not None:
        y = X_df.pop(y_col).values
    else:
        y = None

    X = {}
    for col in X_df.columns:
        values = X_df[col].values
        if isinstance(values[0], (list, np.ndarray)):
            values = np.array([l for l in values], dtype=np.float32)
        X[col] = values

    return lambda: _dataset(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
    )


def _dataset(x, y=None, batch_size=128, num_epochs=1, shuffle=False, seed=None):
    if y is None:
        dataset = tf.data.Dataset.from_tensor_slices(x)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

    if shuffle:
        dataset = dataset.shuffle(
            1000, seed=seed, reshuffle_each_iteration=True  # buffer size = 1000
        )
    elif seed is not None:
        import warnings

        warnings.warn("Seed was set but `shuffle=False`. Seed will be ignored.")

    return dataset.repeat(num_epochs).batch(batch_size)


def build_optimizer(name, lr=0.001, **kwargs):
    """Get an optimizer for TensorFlow high-level API Estimator.

    Args:
        name (str): Optimizer name. Note, to use 'Momentum', should specify
        lr (float): Learning rate
        kwargs: Optimizer arguments as key-value pairs

    Returns:
        tf.train.Optimizer
    """
    optimizers = dict(
        adadelta=tf.train.AdadeltaOptimizer,
        adagrad=tf.train.AdagradDAOptimizer,
        adam=tf.train.AdamOptimizer,
        ftrl=tf.train.FtrlOptimizer,
        momentum=tf.train.MomentumOptimizer,
        rmsprop=tf.train.RMSPropOptimizer,
        sgd=tf.train.GradientDescentOptimizer,
    )

    try:
        optimizer_class = optimizers[name.lower()]
    except KeyError:
        raise KeyError(
            "Optimizer name should be one of: [{}]".format(", ".join(optimizers.keys()))
        )

    # assign default values
    if name.lower() == "momentum" and "momentum" not in kwargs:
        kwargs["momentum"] = 0.9

    return optimizer_class(learning_rate=lr, **kwargs)


def evaluation_log_hook(
    estimator,
    logger,
    true_df,
    y_col,
    eval_df,
    every_n_iter=10000,
    model_dir=None,
    batch_size=256,
    eval_fns=None,
    **eval_kwargs
):
    """Evaluation log hook for TensorFlow high-levmodel_direl API Estimator.
    Note, to evaluate the model in the middle of training (by using this hook),
    the model checkpointing steps should be equal or larger than the hook's since
    TensorFlow Estimator uses the last checkpoint for evaluation or prediction.
    Checkpoint frequency can be set via Estimator's run config.

    Args:
        estimator (tf.estimator.Estimator): Model to evaluate.
        logger (Logger): Custom logger to log the results. E.g., define a subclass of Logger for AzureML logging.
        true_df (pd.DataFrame): Ground-truth data.
        y_col (str): Label column name in true_df
        eval_df (pd.DataFrame): Evaluation data. May not include the label column as
            some evaluation functions do not allow.
        every_n_iter (int): Evaluation frequency (steps). Should be equal or larger than checkpointing steps.
        model_dir (str): Model directory to save the summaries to. If None, does not record.
        batch_size (int): Number of samples fed into the model at a time.
            Note, the batch size doesn't affect on evaluation results.
        eval_fns (iterable of functions): List of evaluation functions that have signature of
            (true_df, prediction_df, **eval_kwargs)->(float). If None, loss is calculated on true_df.
        **eval_kwargs: Evaluation function's keyword arguments. Note, prediction column name should be 'prediction'

    Returns:
        tf.train.SessionRunHook: Session run hook to evaluate the model while training.
    """

    return _TrainLogHook(
        estimator,
        logger,
        true_df,
        y_col,
        eval_df,
        every_n_iter,
        model_dir,
        batch_size,
        eval_fns,
        **eval_kwargs
    )


class MetricsLogger:
    def __init__(self):
        """Log metrics. Each metric's log will be stored in the corresponding list."""
        self._log = {}

    def log(self, metric, value):
        if metric not in self._log:
            self._log[metric] = []
        self._log[metric].append(value)

    def get_log(self):
        return self._log


class _TrainLogHook(tf.train.SessionRunHook):
    def __init__(
        self,
        estimator,
        logger,
        true_df,
        y_col,
        eval_df,
        every_n_iter=1000,
        model_dir=None,
        batch_size=256,
        eval_fns=None,
        **eval_kwargs
    ):
        """Evaluation log hook class"""
        self.model = estimator
        self.logger = logger
        self.true_df = true_df
        self.y_col = y_col
        self.eval_df = eval_df
        self.every_n_iter = every_n_iter
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.eval_fns = eval_fns
        self.eval_kwargs = eval_kwargs

        self.summary_writer = None
        self.global_step_tensor = None
        self.step = 0

    def begin(self):
        if self.model_dir is not None:
            self.summary_writer = tf.summary.FileWriterCache.get(self.model_dir)
            self.global_step_tensor = tf.train.get_or_create_global_step()
        else:
            self.step = 0

    def before_run(self, run_context):
        if self.global_step_tensor is not None:
            requests = {"global_step": self.global_step_tensor}
            return tf.train.SessionRunArgs(requests)
        else:
            return None

    def after_run(self, run_context, run_values):
        if self.global_step_tensor is not None:
            self.step = run_values.results["global_step"]
        else:
            self.step += 1

        if self.step > 1 and self.step % self.every_n_iter == 0:
            _prev_log_level = tf.logging.get_verbosity()
            tf.logging.set_verbosity(tf.logging.ERROR)

            if self.eval_fns is None:
                result = self.model.evaluate(
                    input_fn=pandas_input_fn(
                        df=self.true_df, y_col=self.y_col, batch_size=self.batch_size
                    )
                )["average_loss"]
                self._log("validation_loss", result)
            else:
                predictions = list(
                    itertools.islice(
                        self.model.predict(
                            input_fn=pandas_input_fn(
                                df=self.eval_df, batch_size=self.batch_size
                            )
                        ),
                        len(self.eval_df),
                    )
                )
                prediction_df = self.eval_df.copy()
                prediction_df["prediction"] = [p["predictions"][0] for p in predictions]
                for fn in self.eval_fns:
                    result = fn(self.true_df, prediction_df, **self.eval_kwargs)
                    self._log(fn.__name__, result)

            tf.logging.set_verbosity(_prev_log_level)

    def end(self, session):
        if self.summary_writer is not None:
            self.summary_writer.flush()

    def _log(self, tag, value):
        self.logger.log(tag, value)
        if self.summary_writer is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
            self.summary_writer.add_summary(summary, self.step)
