# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import collections
import itertools

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.estimator.inputs.queues import feeding_functions
from tensorflow.python.estimator.inputs.numpy_io import (
    _get_unique_target_key,
    _validate_and_convert_features,
)

MODEL_DIR = 'model_checkpoints'


def pandas_input_fn(
    df,
    y_col=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
    seed=None,
    num_threads=1
):
    """Pandas input function for TensorFlow high-level API Estimator.

    tf.estimator.inputs.pandas_input_fn cannot handle array/list column properly.
    If the df does not include any array/list data column, one can simply use TensorFlow's pandas_input_fn.

    For more information, see (https://www.tensorflow.org/api_docs/python/tf/estimator/inputs/numpy_input_fn)

    Args:
        df (pd.DataFrame): Data containing features.
        y_col (str): Label column name if df has it.
        batch_size (int): Batch size for the input function.
        num_epochs (int): Number of epochs to iterate over data. If None will run forever.
        shuffle (bool): If True, shuffles the data queue.
        seed (int): Random seed for shuffle.
        num_threads (int): Number of threads used for reading and enqueueing.

    Returns:
        Function that has signature of ()->(dict of features, targets)
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

    return numpy_input_fn(
        x=X,
        y=y,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        seed=seed,
        num_threads=num_threads
    )


def numpy_input_fn(
    x,
    y=None,
    batch_size=128,
    num_epochs=1,
    shuffle=False,
    seed=None,
    queue_capacity=1000,
    num_threads=1
):
    """Numpy input function for TensorFlow high-level API Estimator.
    
    This function is exactly the same as tensorflow's (tf.estimator.inputs.numpy_input_fn),
    except this one allows the seed for shuffle.
        
    Args:
        x (dict): Dictionary of feature (numpy array) object.
        y (np.array): Labels. `None` if absent.
        batch_size (int): Size of batches to return.
        num_epochs (int): Number of epochs to iterate over data. If `None` will run forever.
        shuffle (bool): If True shuffles the queue. Avoid shuffle at prediction time.
        seed (int): Random seed for shuffle.
        queue_capacity (int): Size of queue to accumulate.
        num_threads (int): Number of threads used for reading and enqueueing.
            In order to get deterministic results, use `num_threads=1`.
        
    Returns:
        Function that has signature of ()->(dict of features, targets)
    """
    def input_fn():
        # Note that `x` should not be used after conversion to ordered_dict_data,
        # as type could be either dict or array.
        ordered_dict_data = _validate_and_convert_features(x)

        # Deep copy keys which is a view in python 3
        feature_keys = list(ordered_dict_data.keys())

        if y is None:
            target_keys = None
        elif isinstance(y, dict):
            if not y:
                raise ValueError('y cannot be empty dict, use None instead.')

            ordered_dict_y = collections.OrderedDict(
                sorted(y.items(), key=lambda t: t[0]))
            target_keys = list(ordered_dict_y.keys())

            duplicate_keys = set(feature_keys).intersection(set(target_keys))
            if duplicate_keys:
                raise ValueError('{} duplicate keys are found in both x and y: '
                                 '{}'.format(len(duplicate_keys), duplicate_keys))

            ordered_dict_data.update(ordered_dict_y)
        else:
            target_keys = _get_unique_target_key(ordered_dict_data)
            ordered_dict_data[target_keys] = y

        if len(set(v.shape[0] for v in ordered_dict_data.values())) != 1:
            shape_dict_of_x = {k: ordered_dict_data[k].shape for k in feature_keys}

            if target_keys is None:
                shape_of_y = None
            elif isinstance(target_keys, str):
                shape_of_y = y.shape
            else:
                shape_of_y = {k: ordered_dict_data[k].shape for k in target_keys}

            raise ValueError('Length of tensors in x and y is mismatched. All '
                             'elements in x and y must have the same length.\n'
                             'Shapes in x: {}\n'
                             'Shapes in y: {}\n'.format(shape_dict_of_x, shape_of_y))

        queue = feeding_functions._enqueue_data(  # pylint: disable=protected-access
            ordered_dict_data,
            queue_capacity,
            shuffle=shuffle,
            seed=seed,
            num_threads=num_threads,
            enqueue_size=batch_size,
            num_epochs=num_epochs)

        batch = (
            queue.dequeue_many(batch_size)
            if num_epochs is None else queue.dequeue_up_to(batch_size))

        # Remove the first `Tensor` in `batch`, which is the row number.
        if batch:
            batch.pop(0)

        if isinstance(x, np.ndarray):
            # Return as the same type as original array.
            features = batch[0]
        else:
            # Return as the original dict type
            features = dict(zip(feature_keys, batch[:len(feature_keys)]))

        if target_keys is None:
            return features
        elif isinstance(target_keys, str):
            target = batch[-1]
            return features, target
        else:
            target = dict(zip(target_keys, batch[-len(target_keys):]))
            return features, target

    return input_fn


def build_optimizer(name, lr=0.001, **kwargs):
    """Get an optimizer for TensorFlow high-level API Estimator.

    Args:
        name (str): Optimizer name. Note, to use 'Momentum', should specify
        lr (float): Learning rate.
        kwargs (dictionary): Optimizer arguments.

    Returns:
        tf.train.Optimizer
    """
    if name == 'Adadelta':
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, **kwargs)
    elif name == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr, **kwargs)
    elif name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, **kwargs)
    elif name == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=lr, **kwargs)
    elif name == 'Momentum':
        if 'momentum' in kwargs:
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, **kwargs)
        else:
            optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, **kwargs)
    elif name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, **kwargs)
    elif name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, **kwargs)

    else:
        raise ValueError(
            """Optimizer name should be either 'Adadelta', 'Adagrad', 'Adam',
            'Ftrl', 'Momentum', 'RMSProp', or 'SGD'"""
        )

    return optimizer


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
            _prev_log_level = tf.logging.get_verbosity()
            tf.logging.set_verbosity(tf.logging.ERROR)

            if self.eval_fns is None:
                result = self.model.evaluate(
                    input_fn=pandas_input_fn(
                        df=self.true_df,
                        y_col=self.y_col,
                        batch_size=self.batch_size,
                    )
                )['average_loss']
                self._log('validation_loss', result)
            else:
                predictions = list(itertools.islice(
                    self.model.predict(input_fn=pandas_input_fn(
                        df=self.eval_df,
                        batch_size=self.batch_size,
                    )),
                    len(self.eval_df)
                ))
                prediction_df = self.eval_df.copy()
                prediction_df['prediction'] = [p['predictions'][0] for p in predictions]
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
            summary = tf.Summary(
                value=[tf.Summary.Value(tag=tag, simple_value=value)]
            )
            self.summary_writer.add_summary(summary, self.step)
