# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf

from recommenders.utils.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL
from recommenders.utils.tf_utils import MODEL_DIR


def build_feature_columns(
    users,
    items,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    item_feat_col=None,
    crossed_feat_dim=1000,
    user_dim=8,
    item_dim=8,
    item_feat_shape=None,
    model_type="wide_deep",
):
    """Build wide and/or deep feature columns for TensorFlow high-level API Estimator.

    Args:
        users (iterable): Distinct user ids.
        items (iterable): Distinct item ids.
        user_col (str): User column name.
        item_col (str): Item column name.
        item_feat_col (str): Item feature column name for 'deep' or 'wide_deep' model.
        crossed_feat_dim (int): Crossed feature dimension for 'wide' or 'wide_deep' model.
        user_dim (int): User embedding dimension for 'deep' or 'wide_deep' model.
        item_dim (int): Item embedding dimension for 'deep' or 'wide_deep' model.
        item_feat_shape (int or an iterable of integers): Item feature array shape for 'deep' or 'wide_deep' model.
        model_type (str): Model type, either
            'wide' for a linear model,
            'deep' for a deep neural networks, or
            'wide_deep' for a combination of linear model and neural networks.

    Returns:
        list, list:
        - The wide feature columns
        - The deep feature columns. If only the wide model is selected, the deep column list is empty and viceversa.
    """
    if model_type not in ["wide", "deep", "wide_deep"]:
        raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")

    user_ids = tf.feature_column.categorical_column_with_vocabulary_list(
        user_col, users
    )
    item_ids = tf.feature_column.categorical_column_with_vocabulary_list(
        item_col, items
    )

    if model_type == "wide":
        return _build_wide_columns(user_ids, item_ids, crossed_feat_dim), []
    elif model_type == "deep":
        return (
            [],
            _build_deep_columns(
                user_ids, item_ids, user_dim, item_dim, item_feat_col, item_feat_shape
            ),
        )
    elif model_type == "wide_deep":
        return (
            _build_wide_columns(user_ids, item_ids, crossed_feat_dim),
            _build_deep_columns(
                user_ids, item_ids, user_dim, item_dim, item_feat_col, item_feat_shape
            ),
        )


def _build_wide_columns(user_ids, item_ids, hash_bucket_size=1000):
    """Build wide feature (crossed) columns. `user_ids` * `item_ids` are hashed into `hash_bucket_size`

    Args:
        user_ids (tf.feature_column.categorical_column_with_vocabulary_list): User ids.
        item_ids (tf.feature_column.categorical_column_with_vocabulary_list): Item ids.
        hash_bucket_size (int): Hash bucket size.

    Returns:
        list: Wide feature columns.
    """
    # Including the original features in addition to the crossed one is recommended to address hash collision problem.
    return [
        user_ids,
        item_ids,
        tf.feature_column.crossed_column(
            [user_ids, item_ids], hash_bucket_size=hash_bucket_size
        ),
    ]


def _build_deep_columns(
    user_ids, item_ids, user_dim, item_dim, item_feat_col=None, item_feat_shape=1
):
    """Build deep feature columns

    Args:
        user_ids (tf.feature_column.categorical_column_with_vocabulary_list): User ids.
        item_ids (tf.feature_column.categorical_column_with_vocabulary_list): Item ids.
        user_dim (int): User embedding dimension.
        item_dim (int): Item embedding dimension.
        item_feat_col (str): Item feature column name.
        item_feat_shape (int or an iterable of integers): Item feature array shape.

    Returns:
        list: Deep feature columns.
    """
    deep_columns = [
        # User embedding
        tf.feature_column.embedding_column(
            categorical_column=user_ids, dimension=user_dim, max_norm=user_dim ** 0.5
        ),
        # Item embedding
        tf.feature_column.embedding_column(
            categorical_column=item_ids, dimension=item_dim, max_norm=item_dim ** 0.5
        ),
    ]
    # Item feature
    if item_feat_col is not None:
        deep_columns.append(
            tf.feature_column.numeric_column(
                item_feat_col, shape=item_feat_shape, dtype=tf.float32
            )
        )
    return deep_columns


def build_model(
    model_dir=MODEL_DIR,
    wide_columns=(),
    deep_columns=(),
    linear_optimizer="Ftrl",
    dnn_optimizer="Adagrad",
    dnn_hidden_units=(128, 128),
    dnn_dropout=0.0,
    dnn_batch_norm=True,
    log_every_n_iter=1000,
    save_checkpoints_steps=10000,
    seed=None,
):
    """Build wide-deep model.

    To generate wide model, pass wide_columns only.
    To generate deep model, pass deep_columns only.
    To generate wide_deep model, pass both wide_columns and deep_columns.

    Args:
        model_dir (str): Model checkpoint directory.
        wide_columns (list of tf.feature_column): Wide model feature columns.
        deep_columns (list of tf.feature_column): Deep model feature columns.
        linear_optimizer (str or tf.train.Optimizer): Wide model optimizer name or object.
        dnn_optimizer (str or tf.train.Optimizer): Deep model optimizer name or object.
        dnn_hidden_units (list of int): Deep model hidden units. E.g., [10, 10, 10] is three layers of 10 nodes each.
        dnn_dropout (float): Deep model's dropout rate.
        dnn_batch_norm (bool): Deep model's batch normalization flag.
        log_every_n_iter (int): Log the training loss for every n steps.
        save_checkpoints_steps (int): Model checkpoint frequency.
        seed (int): Random seed.

    Returns:
        tf.estimator.Estimator: Model
    """
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.gpu_options.allow_growth = True # dynamic memory allocation

    # TensorFlow training setup
    config = tf.estimator.RunConfig(
        tf_random_seed=seed,
        log_step_count_steps=log_every_n_iter,
        save_checkpoints_steps=save_checkpoints_steps,
        session_config=gpu_config,
    )

    if len(wide_columns) > 0 and len(deep_columns) == 0:
        model = tf.compat.v1.estimator.LinearRegressor(
            model_dir=model_dir,
            config=config,
            feature_columns=wide_columns,
            optimizer=linear_optimizer,
        )
    elif len(wide_columns) == 0 and len(deep_columns) > 0:
        model = tf.compat.v1.estimator.DNNRegressor(
            model_dir=model_dir,
            config=config,
            feature_columns=deep_columns,
            hidden_units=dnn_hidden_units,
            optimizer=dnn_optimizer,
            dropout=dnn_dropout,
            batch_norm=dnn_batch_norm,
        )
    elif len(wide_columns) > 0 and len(deep_columns) > 0:
        model = tf.compat.v1.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            config=config,
            # wide settings
            linear_feature_columns=wide_columns,
            linear_optimizer=linear_optimizer,
            # deep settings
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=dnn_hidden_units,
            dnn_optimizer=dnn_optimizer,
            dnn_dropout=dnn_dropout,
            batch_norm=dnn_batch_norm,
        )
    else:
        raise ValueError(
            "To generate wide model, set wide_columns.\n"
            "To generate deep model, set deep_columns.\n"
            "To generate wide_deep model, set both wide_columns and deep_columns."
        )

    return model
