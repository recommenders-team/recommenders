# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd


def pandas_input_fn(
    df,
    y_col=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
    num_threads=1
):
    """Pandas input function for TensorFlow high-level API Estimator.

    tf.estimator.inputs.pandas_input_fn cannot handle array/list column properly.
    If the df does not include any array/list data column, one can simply use TensorFlow's pandas_input_fn.

    For more information, see (https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn)

    Args:
        df (pd.DataFrame): Data containing features
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function
        num_epochs (int): Number of number of epochs to iterate over data. If None will run forever.
        shuffle (bool): If True, shuffles the data queue.
        num_threads (int): Number of threads used for reading and enqueueing.

    Returns:
        (tf.estimator.inputs.numpy_input_fn) function that has signature of ()->(dict of features, targets)
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

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        num_threads=num_threads
    )

    return input_fn


def evaluation_log_hook(
    estimator,
    true_df,
    y_col,
    eval_df,
    every_n_iter=1000,
    model_dir=None,
    eval_fn=None,
    **eval_kwargs
):
    """Evaluation log hook for TensorFlow high-levmodel_direl API Estimator.

    Args:
        estimator (tf.estimator.Estimator): Model to evaluate.
        true_df (pd.DataFrame): Ground-truth data.
        y_col (str): Label column name in true_df
        eval_df (pd.DataFrame): Evaluation data. May not include the label column as
            some evaluation functions do not allow.
        every_n_iter (int): Evaluation frequency (steps).
        model_dir (str): Model directory to save the summaries to. If None, does not record.
        eval_fn (function): Evaluation function that has signature of
            (true_df, prediction_df, **eval_kwargs)->(float). If None, loss is calculated on true_df.
        **eval_kwargs: Evaluation function's keyword arguments. Note, prediction column name should be 'prediction'

    Returns:
        (tf.train.SessionRunHook) Session run hook to evaluate the model while training.
    """

    return _TrainLogHook(
        estimator,
        true_df,
        y_col,
        eval_df,
        every_n_iter,
        model_dir,
        eval_fn,
        **eval_kwargs
    )


class _TrainLogHook(tf.train.SessionRunHook):
    def __init__(
        self,
        estimator,
        true_df,
        y_col,
        eval_df,
        every_n_iter=1000,
        model_dir=None,
        eval_fn=None,
        **eval_kwargs
    ):
        """Evaluation log hook class"""
        self.model = estimator
        self.true_df = true_df
        self.y_col = y_col
        self.eval_df = eval_df
        self.every_n_iter = every_n_iter
        self.model_dir = model_dir
        self.eval_fn = eval_fn
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
            requests = {'global_step': self.global_step_tensor}
            return tf.train.SessionRunArgs(requests)
        else:
            return None

    def after_run(self, run_context, run_values):
        if self.global_step_tensor is not None:
            self.step = run_values.results['global_step']
        else:
            self.step += 1

        if self.step > 1 and self.step % self.every_n_iter == 0:
            eval_log = []

            loss = self.model.evaluate(
                input_fn=pandas_input_fn(
                    df=self.true_df,
                    y_col=self.y_col
                ),
                steps=None
            )['average_loss']

            eval_log.append("average_loss = {}".format(loss))
            self._write_simple_value_summary('evaluation/average_loss', loss)

            if self.eval_fn is not None:
                predictions = list(itertools.islice(
                    self.model.predict(input_fn=pandas_input_fn(
                        df=self.eval_df,
                        batch_size=10000,
                    )),
                    len(self.eval_df)
                ))

                prediction_df = self.eval_df.copy()
                prediction_df['prediction'] = [p['predictions'][0] for p in predictions]
                result = self.eval_fn(self.true_df, prediction_df, **self.eval_kwargs)

                eval_log.append("{0} = {1:.5f}".format(self.eval_fn.__name__, result))
                self._write_simple_value_summary('evaluation/'+self.eval_fn.__name__, result)

            eval_log.append("step = {}".format(self.step))
            tf.logging.info("Evaluation: " + ", ".join(eval_log))

    def end(self, session):
        if self.summary_writer is not None:
            self.summary_writer.flush()

    def _write_simple_value_summary(self, tag, value):
        if self.summary_writer is not None:
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)]
            )
            self.summary_writer.add_summary(summary, self.step)
