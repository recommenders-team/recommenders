# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import argparse
import itertools

import pandas as pd
import numpy as np

import tensorflow as tf
from azureml.core import Run

try:
    from reco_utils.aml.tf_log_hook import AmlTfLogHook
    from reco_utils.common import tf_utils
    from reco_utils.evaluation.python_evaluation import (
            rmse, mae, rsquared, exp_var,
            map_at_k, ndcg_at_k, precision_at_k, recall_at_k
    )
except ModuleNotFoundError:
    # When upload to the cloud compute
    from aml.tf_log_hook import AmlTfLogHook
    from common import tf_utils
    from evaluation.python_evaluation import (
        rmse, mae, rsquared, exp_var,
        map_at_k, ndcg_at_k, precision_at_k, recall_at_k
    )


print("TensorFlow version:", tf.VERSION, sep="\n")

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--datastore', type=str, dest='datastore', help="Datastore path")
parser.add_argument('--train-datapath', type=str, dest='train_datapath')
parser.add_argument('--eval-datapath', type=str, dest='eval_datapath')
# Data column names
parser.add_argument('--user-col', type=str, dest='user_col', default='UserId')
parser.add_argument('--item-col', type=str, dest='item_col', default='ItemId')
parser.add_argument('--rating-col', type=str, dest='rating_col', default='Rating')
# Optional feature columns. If not provided, not used
parser.add_argument('--timestamp-col', type=str, dest='timestamp_col')
parser.add_argument('--item-feat-col', type=str, dest='item_feat_col')
# Model type: either 'wide', 'deep', or 'wide_deep'
parser.add_argument('--model-type', type=str, dest='model_type', default='wide_deep')
# Wide model params
parser.add_argument('--linear-optimizer', type=str, dest='linear_optimizer', default='Ftrl')
parser.add_argument('--linear-optimizer-lr', type=float, dest='linear_optimizer_lr', default=0.1)
# Deep model params
parser.add_argument('--dnn-optimizer', type=str, dest='dnn_optimizer', default='Adagrad')
parser.add_argument('--dnn-optimizer-lr', type=float, dest='dnn_optimizer_lr', default=0.1)
parser.add_argument('--dnn-hidden-units', type=str, dest='dnn_hidden_units', default="256,256,128")
parser.add_argument('--dnn-user-embedding-dim', type=int, dest='dnn_user_embedding_dim', default=4)
parser.add_argument('--dnn-item-embedding-dim', type=int, dest='dnn_item_embedding_dim', default=5)

parser.add_argument('--dnn-batch-norm', type=bool, dest='dnn_batch_norm', default=False)
# Training parameters
parser.add_argument('--batch-size', type=int, dest='batch_size', default=256)
parser.add_argument('--epochs', type=int, dest='epochs', default=20)
parser.add_argument('--metrics-list', type=str, nargs='*', dest='metrics_list', default=['RMSE'])

args = parser.parse_args()

# Model checkpoints folder
MODEL_DIR = './models'

MODEL_TYPE = args.model_type
if MODEL_TYPE not in {'wide', 'deep', 'wide_deep'}:
    raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")

BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
METRICS_LIST = args.metrics_list

# Features
USER_COL = args.user_col
ITEM_COL = args.item_col
RATING_COL = args.rating_col
ITEM_FEAT_COL = args.item_feat_col  # e.g. genres, as a list of 0 or 1 (a movie may have multiple genres)
TIMESTAMP_COL = args.timestamp_col
PREDICTION_COL = 'prediction'

# Recommendation evaluator columns
cols = {
    'col_user': USER_COL,
    'col_item': ITEM_COL,
    'col_rating': RATING_COL,
    'col_prediction': PREDICTION_COL,
}

# Wide model hyperparameters
LINEAR_OPTIMIZER = args.linear_optimizer
LINEAR_OPTIMIZER_LR = args.linear_optimizer_lr
# Deep model hyperparameters
DNN_OPTIMIZER = args.dnn_optimizer
DNN_OPTIMIZER_LR = args.dnn_optimizer_lr
DNN_USER_DIM = args.dnn_user_embedding_dim
DNN_ITEM_DIM = args.dnn_item_embedding_dim
DNN_HIDDEN_UNITS = [int(l) for l in args.dnn_hidden_units.split(',')]
DNN_BATCH_NORM = args.dnn_batch_norm

# Load data
X_train = pd.read_pickle(path=os.path.join(args.datastore, args.train_datapath))
y_train = X_train.pop(RATING_COL)

rate_eval = pd.read_pickle(path=os.path.join(args.datastore, args.eval_datapath))
X_rate_eval = rate_eval.copy()
y_rate_eval = X_rate_eval.pop(RATING_COL)

# Get full list of users and items (movies)
user_list = np.unique(
    np.concatenate((X_train[USER_COL].unique(), X_rate_eval[USER_COL].unique()), axis=None)
)
item_list = np.unique(
    np.concatenate((X_train[ITEM_COL].unique(), X_rate_eval[ITEM_COL].unique()), axis=None)
)

# Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
user_item_col = [USER_COL, ITEM_COL]
user_item_list = list(itertools.product(user_list, item_list))
users_items = pd.DataFrame(user_item_list, columns=user_item_col)
# Remove seen items (items in the train set)
users_items_exclude_train = users_items.loc[
    ~users_items.set_index(user_item_col).index.isin(X_train.set_index(user_item_col).index)
]
# Add timestamp info
X_rank_eval = pd.merge(X_rate_eval, users_items_exclude_train, on=user_item_col, how='outer')
X_rank_eval.fillna(X_rate_eval[TIMESTAMP_COL].max(), inplace=True)
# TODO add other columns too if needed

# Get AML run context
root_run = Run.get_context()

arguments = str(vars(args))
print("Args:", arguments, sep='\n')
root_run.log("Args", arguments)

# Exhaustive search for regularization param to see how it affects
if MODEL_TYPE == 'deep' or MODEL_TYPE == 'wide_deep':
    reg_name = 'dropout'
    regs = np.linspace(0.0, 0.5, 5)
elif LINEAR_OPTIMIZER == 'Ftrl':
    reg_name = 'l1_reg'
    regs = np.linspace(0.0, 0.1, 5)
else:
    reg_name = None
    regs = [0.0]

for reg in regs:
    if reg_name is not None:
        print("{}: {}".format(reg_name, reg))
        root_run.log(reg_name, reg)

    # Feature columns
    wide_columns = []
    deep_columns = []
    if MODEL_TYPE == 'wide' or MODEL_TYPE == 'wide_deep':
        wide_columns = tf_utils.build_feature_columns(
            'wide', user_list, item_list, 'UserId', 'MovieId'
        )
    if MODEL_TYPE == 'deep' or MODEL_TYPE == 'wide_deep':
        deep_columns = tf_utils.build_feature_columns(
            'deep', user_list, item_list, 'UserId', 'MovieId', None, 'Timestamp',
            DNN_USER_DIM, DNN_ITEM_DIM, 0
        )

    # Model (Estimator). Note, if you want an Estimator optimized for a specific metrics, write a custom one.
    if MODEL_TYPE == 'wide':
        model = tf.estimator.LinearRegressor(  # LinearClassifier(
            model_dir=MODEL_DIR,
            feature_columns=wide_columns,
            optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR, ftrl_l1_reg=reg)
        )
    elif MODEL_TYPE == 'deep':
        model = tf.estimator.DNNRegressor(  # DNNClassifier(
            model_dir=MODEL_DIR,
            feature_columns=deep_columns,
            hidden_units=DNN_HIDDEN_UNITS,
            optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR),
            dropout=reg,
            batch_norm=DNN_BATCH_NORM
        )
    elif MODEL_TYPE == 'wide_deep':
        model = tf.estimator.DNNLinearCombinedRegressor(  # DNNLinearCombinedClassifier(
            model_dir=MODEL_DIR,
            # wide settings
            linear_feature_columns=wide_columns,
            linear_optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR, ftrl_l1_reg=reg),
            # deep settings
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR),
            dnn_dropout=reg,
            batch_norm=DNN_BATCH_NORM
        )
    else:
        # This should not happen
        model = None

    # start an Azure ML run
    # with root_run.child_run("reg-{}".format(reg)) as run:
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        x=X_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=True,
        num_threads=1
    )
    model = tf.contrib.estimator.add_metrics(model, tf_utils.eval_metrics('mae'))
    # aml_hook = AmlTfLogHook(
    #     aml_run=run,
    #     model=model,
    #     X_eval=X_rate_eval,
    #     y_eval=y_rate_eval,
    #     metrics=['loss', 'mae'],
    #     every_n_iter=200
    # )
    try:
        model.train(input_fn=train_input_fn)  #, hooks=[aml_hook])

        # TODO maybe drop the cold users and items

        # TODO for now, just ndcg and rmse
        if 'NDCG' in METRICS_LIST:
            predictions = list(model.predict(
                input_fn=tf.estimator.inputs.pandas_input_fn(
                    x=X_rank_eval,
                    batch_size=100,
                    num_epochs=1,
                    shuffle=False
                )
            ))
            reco = X_rank_eval.copy()
            reco[PREDICTION_COL] = pd.Series([p['predictions'][0] for p in predictions]).values

            # TODO for now, fix TOP_K
            TOP_K = 10
            eval_ndcg = ndcg_at_k(rate_eval, reco, k=TOP_K, **cols)

            print("NDCG:", eval_ndcg)
            root_run.log("NDCG", eval_ndcg)

        if 'RMSE' in METRICS_LIST:
            predictions = list(model.predict(
                input_fn=tf.estimator.inputs.pandas_input_fn(
                    x=X_rate_eval,
                    batch_size=100,
                    num_epochs=1,
                    shuffle=False
                )
            ))
            rate = X_rate_eval.copy()
            rate[PREDICTION_COL] = pd.Series([p['predictions'][0] for p in predictions]).values

            eval_rmse = rmse(rate_eval, rate, **cols)

            print("RMSE:", eval_rmse)
            root_run.log("RMSE", eval_rmse)

    except tf.train.NanLossDuringTrainingError as e:
        print(e.message)




# Save model. Note, AML automatically upload the files saved in the "./outputs" folder into run history
# MODEL_OUTPUT_DIR = './outputs/model'
# os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
#
# feature_spec = tf.feature_column.make_parse_example_spec(col)
# export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)
#
# probablly save checkpoint or export model

# TODO clean up MODEL_DIR and MODEL_OUTPUT_DIR


"""
For details of examples of export and load models,
https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md
https://www.tensorflow.org/guide/saved_model
https://github.com/monk1337/DNNClassifier-example/
"""
