import tensorflow as tf


def build_feature_columns(
    feat_type,
    user_list,
    item_list,
    user_col,
    item_col,
    item_feat_col=None,
    timestamp_col=None,
    user_dim=0,
    item_dim=0,
    item_feat_num=0
):
    """
    In case of deep, convert categorical features into a lower-dimensional embedding vectors

    :param feat_type: 'wide' or 'deep'
    :param user_list: Distinct user list
    :param item_list: Distinct item list
    :param user_col:
    :param item_col:
    :param item_feat_col: Not use if None
    :param timestamp_col: Not use if None
    :param user_dim: Use when the column_type is 'deep'
    :param item_dim: Use when the column_type is 'deep'
    :param item_feat_num: Use when the column_type is 'deep' and item_feat_col is not None
    :return:
    """

    user_id = tf.feature_column.categorical_column_with_vocabulary_list(user_col, user_list)
    item_id = tf.feature_column.categorical_column_with_vocabulary_list(item_col, item_list)

    if feat_type == 'wide':
        # TODO: Maybe include other features, such as...
        # 1. base features including user_id, item_id, genres, and timestamps
        # 2. user - genres cross product
        columns = [
            tf.feature_column.crossed_column([user_col, item_col], hash_bucket_size=1000)
        ]
    elif feat_type == 'deep':
        # User embedding
        user_feat = tf.feature_column.embedding_column(
            categorical_column=user_id,
            dimension=user_dim,
            max_norm=user_dim ** .5
        )
        # Item embedding
        item_feat = tf.feature_column.embedding_column(
            categorical_column=item_id,
            dimension=item_dim,
            max_norm=item_dim ** .5
        )
        columns = [user_feat, item_feat]

        if timestamp_col is not None:
            columns.append(tf.feature_column.numeric_column(timestamp_col))

        if item_feat_col is not None:
            columns.append(
                tf.feature_column.numeric_column(
                    item_feat_col,
                    shape=(item_feat_num,),
                    dtype=tf.uint8
                )
            )
    else:
        raise ValueError("Feature type should be either 'wide' or 'deep'")

    return columns


def build_optimizer(name, lr, ftrl_l1_reg=None):
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


def eval_metrics(name):
    """

    :param name:
    :return:
    """
    if name == 'rmse':
        return lambda labels, predictions: {
            name: tf.metrics.root_mean_squared_error(
                tf.cast(labels, tf.float32),
                predictions['predictions']
            )
        }
    elif name == 'mae':
        return lambda labels, predictions: {
            name: tf.metrics.mean_absolute_error(
                tf.cast(labels, tf.float32),
                predictions['predictions']
            )
        }
    else:
        raise ValueError("Evaluation metrics should be either 'rmse' or 'mae'")
