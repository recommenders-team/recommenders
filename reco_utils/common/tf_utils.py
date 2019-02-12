# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import abc
import itertools
import tensorflow as tf
import numpy as np
import pandas as pd

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL
)


MODEL_DIR = 'model_checkpoints'


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
        num_epochs (int): Number of epochs to iterate over data. If None will run forever.
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


def _build_optimizer(name, lr=0.001, ftrl_l1_reg=0.0):
    """Get an optimizer for TensorFlow high-level API Estimator.

    Args:
        name (str): Optimizer name.
        lr (float): Learning rate.
        ftrl_l1_reg (float): Ftrl optimizer l1 regularization rate. Ignored for other optimizers.

    Returns:
        optimizer (tf.train.Optimizer)
    """
    if name == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    elif name == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    elif name == 'RMSProp':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    elif name == 'SGD':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif name == 'Ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate=lr, l1_regularization_strength=ftrl_l1_reg)
    else:
        raise ValueError("Optimizer type should be either 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', or 'SGD'")

    return optimizer


def _build_wide_columns(user_ids, item_ids):
    """Build wide feature columns

    Args:
        user_ids (tf.feature_column.categorical_column_with_vocabulary_list): User ids.
        item_ids (tf.feature_column.categorical_column_with_vocabulary_list): Item ids.

    Returns:
        (list) Wide feature columns.
    """
    return [
        tf.feature_column.crossed_column([user_ids, item_ids], hash_bucket_size=1000)
    ]


def _build_deep_columns(user_ids, item_ids, user_dim, item_dim,
                        item_feat_col=None, item_feat_shape=1):
    """Build deep feature columns

    Args:
        user_ids (tf.feature_column.categorical_column_with_vocabulary_list): User ids.
        item_ids (tf.feature_column.categorical_column_with_vocabulary_list): Item ids.
        user_dim (int): User embedding dimension.
        item_dim (int): Item embedding dimension.
        item_feat_col (str): Item feature column name.
        item_feat_shape (int or an iterable of integers): Item feature array shape.
    Returns:
        (list) Deep feature columns.
    """
    deep_columns = [
        # User embedding
        tf.feature_column.embedding_column(
            categorical_column=user_ids,
            dimension=user_dim,
            max_norm=user_dim ** .5
        ),
        # Item embedding
        tf.feature_column.embedding_column(
            categorical_column=item_ids,
            dimension=item_dim,
            max_norm=item_dim ** .5
        )
    ]
    # Item feature
    if item_feat_col is not None:
        deep_columns.append(
            tf.feature_column.numeric_column(
                item_feat_col,
                shape=item_feat_shape,
                dtype=tf.float32
            )
        )
    return deep_columns


def build_model(
    users,
    items,
    model_dir=MODEL_DIR,
    model_type='wide_deep',
    linear_optimizer='Ftrl',
    linear_optimizer_lr=0.01,
    linear_l1_reg=0.0,
    dnn_optimizer='Adam',
    dnn_optimizer_lr=0.01,
    dnn_hidden_units=(128, 128),
    dnn_user_dim=8,
    dnn_item_dim=8,
    dnn_dropout=0.0,
    dnn_batch_norm=True,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    item_feat_col=None,
    item_feat_shape=None,
    log_every_n_iter=1000,
    save_checkpoints_steps=10000
):
    """Train and evaluate wide-deep model.

    Args:
        users (iterable): Distinct user ids.
        items (iterable): Distinct item ids.
        model_dir (str): Model checkpoint directory.
        model_type (str): Model type, either
            'wide' for a linear model,
            'deep' for a deep neural networks, or
            'wide_deep' for a combination of linear model and neural networks.
        linear_optimizer (str): Wide model optimizer name.
        linear_optimizer_lr (float): Wide model learning rate.
        linear_l1_reg (float): Ftrl model l1 regularization rate. Will be ignored for other optimizers
        dnn_optimizer (str): Deep model optimizer name.
        dnn_optimizer_lr (float): Deep model learning rate.
        dnn_hidden_units (list of int): Deep model hidden units. E.g., [10, 10, 10] is three layers of 10 nodes each.
        dnn_user_dim (int): User embedding dimension.
        dnn_item_dim (int): Item embedding dimension.
        dnn_dropout (float): Deep model's dropout rate.
        dnn_batch_norm (bool): Deep model's batch normalization flag.
        user_col (str): User column name.
        item_col (str): Item column name.
        item_feat_col (str): Item feature column name.
        item_feat_shape (int or an iterable of integers): Item feature array shape.
        log_every_n_iter (int): Every log_every_n_iter steps, log the train loss.
        save_checkpoints_steps (int): Model checkpointing frequency.

    Returns:
        (tf.estimator.Estimator) model, (list) wide feature columns, (list) deep feature columns.
    """
    # TensorFlow training log frequency setup
    config = tf.estimator.RunConfig(
        log_step_count_steps=log_every_n_iter,
        save_checkpoints_steps=save_checkpoints_steps,
    )

    user_ids = tf.feature_column.categorical_column_with_vocabulary_list(user_col, users)
    item_ids = tf.feature_column.categorical_column_with_vocabulary_list(item_col, items)

    wide_columns = []
    deep_columns = []
    if model_type == 'wide':
        wide_columns = _build_wide_columns(user_ids, item_ids)
        model = tf.estimator.LinearRegressor(
            model_dir=model_dir,
            config=config,
            feature_columns=wide_columns,
            optimizer=_build_optimizer(linear_optimizer, linear_optimizer_lr, linear_l1_reg)
        )
    elif model_type == 'deep':
        deep_columns = _build_deep_columns(user_ids, item_ids, dnn_user_dim, dnn_item_dim, item_feat_col, item_feat_shape)
        model = tf.estimator.DNNRegressor(
            model_dir=model_dir,
            config=config,
            feature_columns=deep_columns,
            hidden_units=dnn_hidden_units,
            optimizer=_build_optimizer(dnn_optimizer, dnn_optimizer_lr),
            dropout=dnn_dropout,
            batch_norm=dnn_batch_norm
        )
    elif model_type == 'wide_deep':
        wide_columns = _build_wide_columns(user_ids, item_ids)
        deep_columns = _build_deep_columns(user_ids, item_ids, dnn_user_dim, dnn_item_dim, item_feat_col, item_feat_shape)
        model = tf.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            config=config,
            # wide settings
            linear_feature_columns=wide_columns,
            linear_optimizer=_build_optimizer(linear_optimizer, linear_optimizer_lr, linear_l1_reg),
            # deep settings
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=dnn_hidden_units,
            dnn_optimizer=_build_optimizer(dnn_optimizer, dnn_optimizer_lr),
            dnn_dropout=dnn_dropout,
            batch_norm=dnn_batch_norm
        )
    else:
        raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")

    return model, wide_columns, deep_columns


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
        (tf.train.SessionRunHook) Session run hook to evaluate the model while training.
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


class Logger(abc.ABC):
    @abc.abstractmethod
    def log(self, tag, value):
        """Custom logger class for evaluation_log_hook

        Args:
            tag (str): tag for the log value. E.g. "rmse"
            value (str): value to log. E.g. "0.03"
        """
        pass


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
