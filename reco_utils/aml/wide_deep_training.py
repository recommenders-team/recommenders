# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
sys.path.append("../../")

import argparse
import itertools
import os
import shutil
import warnings

import pandas as pd
import tensorflow as tf

try:
    from azureml.core import Run
    HAS_AML = True
except ModuleNotFoundError:
    HAS_AML = False

from reco_utils.dataset.pandas_df_utils import user_item_pairs
from reco_utils.dataset.python_splitters import python_random_split
from reco_utils.common import tf_utils
from reco_utils.evaluation.python_evaluation import (
    rmse, mae, rsquared, exp_var,
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)


print("TensorFlow version:", tf.VERSION, sep="\n")

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--datastore', type=str, dest='datastore', help="Datastore path")
parser.add_argument('--train-datapath', type=str, dest='train_datapath')
# Data column names
parser.add_argument('--user-col', type=str, dest='user_col', default='UserId')
parser.add_argument('--item-col', type=str, dest='item_col', default='ItemId')
parser.add_argument('--rating-col', type=str, dest='rating_col', default='Rating')
# Optional feature columns. If not provided, not used
parser.add_argument('--item-feat-col', type=str, dest='item_feat_col')
parser.add_argument('--metrics', type=str, nargs='*', dest='metrics', default=['rmse'])
# Model type: either 'wide', 'deep', or 'wide_deep'
parser.add_argument('--model-type', type=str, dest='model_type', default='wide_deep')
# Wide model params
parser.add_argument('--linear-optimizer', type=str, dest='linear_optimizer', default='Ftrl')
parser.add_argument('--linear-optimizer-lr', type=float, dest='linear_optimizer_lr', default=0.1)
# Deep model params
parser.add_argument('--dnn-optimizer', type=str, dest='dnn_optimizer', default='Adagrad')
parser.add_argument('--dnn-optimizer-lr', type=float, dest='dnn_optimizer_lr', default=0.1)
parser.add_argument('--dnn-hidden-layer-1', type=int, dest='dnn_hidden_layer_1', default=0)
parser.add_argument('--dnn-hidden-layer-2', type=int, dest='dnn_hidden_layer_2', default=0)
parser.add_argument('--dnn-hidden-layer-3', type=int, dest='dnn_hidden_layer_3', default=128)
parser.add_argument('--dnn-hidden-layer-4', type=int, dest='dnn_hidden_layer_4', default=128)
parser.add_argument('--dnn-user-embedding-dim', type=int, dest='dnn_user_embedding_dim', default=8)
parser.add_argument('--dnn-item-embedding-dim', type=int, dest='dnn_item_embedding_dim', default=8)
parser.add_argument('--dnn-batch-norm', type=int, dest='dnn_batch_norm', default=0)
# Training parameters
parser.add_argument('--epochs', type=int, dest='epochs', default=10)
parser.add_argument('--batch-size', type=int, dest='batch_size', default=256)
parser.add_argument('--l1-reg', type=float, dest='l1_reg', default=0.0)
parser.add_argument('--dropout', type=float, dest='dropout', default=0.0)

args = parser.parse_args()

MODEL_TYPE = args.model_type
if MODEL_TYPE not in {'wide', 'deep', 'wide_deep'}:
    raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

# Metrics validity check
RATING_METRICS = set()
RANKING_METRICS = set()
_SUPPORTED_METRICS = {
    'rmse': rmse,
    'mae': mae,
    'rsquared': rsquared,
    'exp_var': exp_var,
    'map': map_at_k,
    'ndcg': ndcg_at_k,
    'precision': precision_at_k,
    'recall': recall_at_k
}
if args.metrics is None:
    raise ValueError(
        """Metrics should be 'rmse', 'mae', 'rsquared', 'exp_var'
            'map@k', 'ndcg@k', 'precision@k', or 'recall@k' where k is a number.
        """
    )

for m in args.metrics:
    name_k = m.split('@')
    if name_k[0] not in _SUPPORTED_METRICS:
        raise ValueError("{} is not a valid metrics name".format(name_k[0]))
    else:
        if len(name_k) == 1:
            RATING_METRICS.add(m)
        else:
            # Check if we have a valid 'top_k' number
            if name_k[1].isdigit():
                RANKING_METRICS.add(m)
            else:
                raise ValueError("{} is not a valid number".format(name_k[1]))
PRIMARY_METRIC = args.metrics[0]

# Features
USER_COL = args.user_col
ITEM_COL = args.item_col
RATING_COL = args.rating_col
ITEM_FEAT_COL = args.item_feat_col  # e.g. genres, as a list of 0 or 1 (a movie may have multiple genres)
PREDICTION_COL = 'prediction'
# Evaluation columns kwargs
cols = {
    'col_user': USER_COL,
    'col_item': ITEM_COL,
    'col_rating': RATING_COL,
    'col_prediction': PREDICTION_COL
}

# Wide model hyperparameters
LINEAR_OPTIMIZER = args.linear_optimizer
LINEAR_OPTIMIZER_LR = args.linear_optimizer_lr
# Deep model hyperparameters
DNN_OPTIMIZER = args.dnn_optimizer
DNN_OPTIMIZER_LR = args.dnn_optimizer_lr
DNN_USER_DIM = args.dnn_user_embedding_dim
DNN_ITEM_DIM = args.dnn_item_embedding_dim
DNN_HIDDEN_UNITS = []
if args.dnn_hidden_layer_1 > 0:
    DNN_HIDDEN_UNITS.append(args.dnn_hidden_layer_1)
if args.dnn_hidden_layer_2 > 0:
    DNN_HIDDEN_UNITS.append(args.dnn_hidden_layer_2)
if args.dnn_hidden_layer_3 > 0:
    DNN_HIDDEN_UNITS.append(args.dnn_hidden_layer_3)
if args.dnn_hidden_layer_4 > 0:
    DNN_HIDDEN_UNITS.append(args.dnn_hidden_layer_4)
DNN_BATCH_NORM = (args.dnn_batch_norm == 1)

L1_REG = args.l1_reg
DROPOUT = args.dropout

LOG_STEPS = 1000
print("Args:", str(vars(args)), sep='\n')

# Get AML run context
if HAS_AML:
    run = Run.get_context()
    run.log('model_type', MODEL_TYPE)
    run.log('linear_optimizer', LINEAR_OPTIMIZER)
    run.log('linear_optimizer_lr', LINEAR_OPTIMIZER_LR)
    run.log('dnn_optimizer', DNN_OPTIMIZER)
    run.log('dnn_optimizer_lr', DNN_OPTIMIZER_LR)
    run.log('dnn_hidden_layer', str(DNN_HIDDEN_UNITS))
    run.log('dnn_user_embedding_dim', DNN_USER_DIM)
    run.log('dnn_item_embedding_dim', DNN_ITEM_DIM)
    run.log('dnn_batch_norm', DNN_BATCH_NORM)
    run.log('batch_size', BATCH_SIZE)
    run.log('l1_reg', L1_REG)
    run.log('dropout', DROPOUT)

if args.datastore is not None:
    data = pd.read_pickle(path=os.path.join(args.datastore, args.train_datapath))
else:
    # For unit testing w/o AML
    from reco_utils.dataset import movielens
    data = movielens.load_pandas_df(
        size='100k',
        header=[USER_COL, ITEM_COL, RATING_COL]
    )

# Unique users and items
if ITEM_FEAT_COL is None:
    items = data.drop_duplicates(ITEM_COL)[[ITEM_COL]].reset_index(drop=True)
else:
    items = data.drop_duplicates(ITEM_COL)[[ITEM_COL, ITEM_FEAT_COL]].reset_index(drop=True)
users = data.drop_duplicates(USER_COL)[[USER_COL]].reset_index(drop=True)

user_id = tf.feature_column.categorical_column_with_vocabulary_list(USER_COL, users[USER_COL].values)
item_id = tf.feature_column.categorical_column_with_vocabulary_list(ITEM_COL, items[ITEM_COL].values)


def aml_wide_deep_training():
    """Train wide and deep model by using the given hyper-parameters
    """
    # Split train and evaluation sets
    train, test = python_random_split(data, ratio=0.75, seed=123)

    # Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
    ranking_pool = user_item_pairs(
        user_df=users,
        item_df=items,
        user_col=USER_COL,
        item_col=ITEM_COL,
        user_item_filter_df=train,
        shuffle=True
    )

    _MODEL_DIR = 'model_checkpoints'
    print(_MODEL_DIR)
    try:
        # Clean-up previous model dir if exists
        shutil.rmtree(_MODEL_DIR)
    except FileNotFoundError:
        pass

    model = _build_model(_MODEL_DIR)

    print("Start training...")
    try:
        model.train(
            input_fn=tf_utils.pandas_input_fn(
                df=train,
                y_col=RATING_COL,
                batch_size=BATCH_SIZE,
                num_epochs=NUM_EPOCHS,
                shuffle=True
            ),
            # Somehow hooks cause DataLossError in AML
        )

        print("Evaluating...")
        if len(RATING_METRICS) > 0:
            predictions = list(itertools.islice(
                model.predict(input_fn=tf_utils.pandas_input_fn(
                    df=test,
                    batch_size=10000,
                )),
                len(test)
            ))
            prediction_df = test.drop(RATING_COL, axis=1)
            prediction_df[PREDICTION_COL] = [p['predictions'][0] for p in predictions]
            for m in RATING_METRICS:
                result = _SUPPORTED_METRICS[m](test, prediction_df, **cols)
                print(m, result)
                if HAS_AML:
                    run.log(m, result)

        if len(RANKING_METRICS) > 0:
            predictions = list(itertools.islice(
                model.predict(input_fn=tf_utils.pandas_input_fn(
                    df=ranking_pool,
                    batch_size=10000,
                )),
                len(ranking_pool)
            ))
            prediction_df = ranking_pool.copy()
            prediction_df[PREDICTION_COL] = [p['predictions'][0] for p in predictions]
            for m in RANKING_METRICS:
                name_k = m.split('@')
                result = _SUPPORTED_METRICS[name_k[0]](test, prediction_df, k=int(name_k[1]), **cols)
                print(m, result)
                if HAS_AML:
                    run.log(m, result)

    except tf.train.NanLossDuringTrainingError:
        warnings.warn("NanLossDuringTrainingError")


def _build_optimizer(name, lr):
    """Get an optimizer for TensorFlow high-level API Estimator.
    """
    if name == 'Adagrad':
        _optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    elif name == 'Adam':
        _optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    elif name == 'RMSProp':
        _optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
    elif name == 'SGD':
        _optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif name == 'Ftrl':
        _optimizer = tf.train.FtrlOptimizer(learning_rate=lr, l1_regularization_strength=L1_REG)
    else:
        raise ValueError("Optimizer type should be either 'Adagrad', 'Adam', 'Ftrl', 'RMSProp', or 'SGD'")

    return _optimizer


def _build_wide_columns():
    """Build wide feature columns
    """
    return [
        tf.feature_column.crossed_column([user_id, item_id], hash_bucket_size=1000)
    ]


def _build_deep_columns():
    """Build deep feature columns
    """
    _deep_columns = [
        # User embedding
        tf.feature_column.embedding_column(
            categorical_column=user_id,
            dimension=DNN_USER_DIM,
            max_norm=DNN_USER_DIM ** .5
        ),
        # Item embedding
        tf.feature_column.embedding_column(
            categorical_column=item_id,
            dimension=DNN_ITEM_DIM,
            max_norm=DNN_ITEM_DIM ** .5
        )
    ]
    # Item feature
    if ITEM_FEAT_COL is not None:
        _deep_columns.append(
            tf.feature_column.numeric_column(
                ITEM_FEAT_COL,
                shape=len(items[ITEM_FEAT_COL][0]),
                dtype=tf.float32
            )
        )
    return _deep_columns


def _build_model(model_dir):
    # Set logging frequency
    tf.logging.set_verbosity(tf.logging.INFO)
    _config = tf.estimator.RunConfig()
    _config = _config.replace(log_step_count_steps=LOG_STEPS)

    if MODEL_TYPE == 'wide':
        return tf.estimator.LinearRegressor(
            model_dir=model_dir,
            config=_config,
            feature_columns=_build_wide_columns(),
            optimizer=_build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR)
        )
    elif MODEL_TYPE == 'deep':
        return tf.estimator.DNNRegressor(
            model_dir=model_dir,
            config=_config,
            feature_columns=_build_deep_columns(),
            hidden_units=DNN_HIDDEN_UNITS,
            optimizer=_build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR),
            dropout=DROPOUT,
            batch_norm=DNN_BATCH_NORM
        )
    elif MODEL_TYPE == 'wide_deep':
        return tf.estimator.DNNLinearCombinedRegressor(
            model_dir=model_dir,
            config=_config,
            # wide settings
            linear_feature_columns=_build_wide_columns(),
            linear_optimizer=_build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR),
            # deep settings
            dnn_feature_columns=_build_deep_columns(),
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_optimizer=_build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR),
            dnn_dropout=DROPOUT,
            batch_norm=DNN_BATCH_NORM
        )
    else:
        raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")


aml_wide_deep_training()
