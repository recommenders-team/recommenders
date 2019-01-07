# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import os
import argparse
import pandas as pd

import tensorflow as tf
from azureml.core import Run

try:
    import tf_utils
except ModuleNotFoundError:
    import reco_utils.common.tf_utils


print("TensorFlow version:", tf.VERSION, "\n")

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--data-folder', type=str, dest='data_folder')
parser.add_argument('--train-set-path', type=str, dest='train_set_path')
parser.add_argument('--eval-set-path', type=str, dest='eval_set_path')
# Data column names
parser.add_argument('--user-col', type=str, dest='user_col', default='UserId')
parser.add_argument('--item-col', type=str, dest='item_col', default='ItemId')
parser.add_argument('--rating-col', type=str, dest='rating_col', default='Rating')
# Optional feature columns. If not provided, not used
parser.add_argument('--item-feat-col', type=str, dest='item_feat_col')
parser.add_argument('--item-feat-num', type=int, dest='item_feat_num')
parser.add_argument('--timestamp-col', type=str, dest='timestamp_col')
# Model type: either 'wide', 'deep', or 'wide_deep'
parser.add_argument('--model-type', type=str, dest='model_type', default='wide_deep')
# Wide model params
parser.add_argument('--linear-optimizer', type=str, dest='linear_optimizer', default='Ftrl')
parser.add_argument('--linear-optimizer-lr', type=float, dest='linear_optimizer_lr', default=0.1)
# Deep model params
parser.add_argument('--user-embedding-dim', type=int, dest='user_embedding_dim', default=4)
parser.add_argument('--item-embedding-dim', type=int, dest='item_embedding_dim', default=5)
parser.add_argument('--hidden-units', type=int, nargs='*', dest='hidden_units', default=[256, 256, 128])
parser.add_argument('--dnn-optimizer', type=str, dest='dnn_optimizer', default='Adagrad')
parser.add_argument('--dnn-optimizer-lr', type=float, dest='dnn_optimizer_lr', default=0.1)
parser.add_argument('--dnn-dropout', type=float, dest='dnn_dropout')
parser.add_argument('--dnn-batch-norm', type=bool, dest='dnn_batch_norm', default=False)
# Training parameters
parser.add_argument('--batch-size', type=int, dest='batch_size', default=256)
parser.add_argument('--epochs', type=int, dest='epochs', default=20)
parser.add_argument('--eval-metrics', type=str, dest='eval_metrics', default='rmse')

args = parser.parse_args()

# Model checkpoints folder TODO should be unique???
# TODO clean out the output directory. Otherwise, it will resume training
MODEL_DIR = './models'

MODEL_TYPE = args.model_type
EVAL_METRICS = args.eval_metrics

# Features
USER_COL = args.user_col
ITEM_COL = args.item_col
RATING_COL = args.rating_col
ITEM_FEAT_COL = args.item_feat_col  # e.g. genres, as a list of 0 or 1 (a movie may have multiple genres)
ITEM_FEAT_NUM = args.item_feat_num  # e.g. number of genres
TIMESTAMP_COL = args.timestamp_col
HIDDEN_UNITS = args.hidden_units

# Hyperparameters
USER_DIM = args.user_embedding_dim
ITEM_DIM = args.item_embedding_dim
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LINEAR_OPTIMIZER = args.linear_optimizer
LINEAR_OPTIMIZER_LR = args.linear_optimizer_lr
DNN_OPTIMIZER = args.dnn_optimizer
DNN_OPTIMIZER_LR = args.dnn_optimizer_lr
DNN_DROPOUT = args.dnn_dropout
DNN_BATCH_NORM = args.dnn_batch_norm

print("Args:")
print(args)

# Load data
X_train = pd.read_pickle(path=os.path.join(args.data_folder, args.train_set_path))
y_train = X_train.pop(RATING_COL)
X_eval = pd.read_pickle(path=os.path.join(args.data_folder, args.eval_set_path))
y_eval = X_eval.pop(RATING_COL)
# TODO

# Hyperparam tuning and evaluation data set split
fit = train.sample(frac=0.75, random_state=123)
evl = train.drop(fit.index)
X_fit = fit.copy()
y_fit = X_fit.pop('Rating')
X_evl = evl.copy()
y_evl = X_evl.pop('Rating')



# Distinct users and items to build features
user_list = X_train[USER_COL].unique()
item_list = X_train[ITEM_COL].unique()
w_col = tf_utils.build_feature_columns(
    'wide', user_list, item_list, USER_COL, ITEM_COL
)
d_col = tf_utils.build_feature_columns(
    'deep', user_list, item_list, USER_COL, ITEM_COL, ITEM_FEAT_COL, TIMESTAMP_COL,
    USER_DIM, ITEM_DIM, ITEM_FEAT_NUM
)

# Model (Estimator). Note, if you want an Estimator optimized for a specific metrics, write a custom one.
if MODEL_TYPE == 'wide':
    model = tf.estimator.LinearRegressor(  # LinearClassifier(
        model_dir=MODEL_DIR,
        feature_columns=w_col,
        optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR)
    )
    col = w_col
elif MODEL_TYPE == 'deep':
    model = tf.estimator.DNNRegressor(  # DNNClassifier(
        model_dir=MODEL_DIR,
        feature_columns=d_col,
        hidden_units=HIDDEN_UNITS,
        optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR),
        dropout=DNN_DROPOUT,
        batch_norm=DNN_BATCH_NORM
    )
    col = d_col
elif MODEL_TYPE == 'wide_deep':
    model = tf.estimator.DNNLinearCombinedRegressor(  # DNNLinearCombinedClassifier(
        model_dir=MODEL_DIR,
        # wide settings
        linear_feature_columns=w_col,
        linear_optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR),
        # deep settings
        dnn_feature_columns=d_col,
        dnn_hidden_units=HIDDEN_UNITS,
        dnn_optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR),
        dnn_dropout=DNN_DROPOUT,
        batch_norm=DNN_BATCH_NORM
    )
    col = w_col + d_col
else:
    raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")

# Add metrics,
model = tf.contrib.estimator.add_metrics(model, tf_utils.eval_metrics(EVAL_METRICS))

train_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    num_epochs=EPOCHS,
    shuffle=True,
    num_threads=1
)

eval_input_fn = tf.estimator.inputs.pandas_input_fn(
    x=X_eval,
    y=y_eval,
    batch_size=BATCH_SIZE,
    num_epochs=1,
    shuffle=False
)

# train_spec = tf.estimator.TrainSpec(
#     input_fn=train_input_fn,
#     max_steps=int(len(X_train)*EPOCHS/float(BATCH_SIZE))  # This will override num_epochs
# )
# eval_spec = tf.estimator.EvalSpec(
#     input_fn=eval_input_fn,
#     steps=None,  # 'None' to evaluate the entire eval set
# )
# tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

model.train(input_fn=train_input_fn)
eval_metrics_val = model.evaluate(input_fn=eval_input_fn)[EVAL_METRICS]

# start an Azure ML run
run = Run.get_context()
# log accuracies
run.log(EVAL_METRICS, eval_metrics_val)

# Save model. Note, AML automatically upload the files saved in the "./outputs" folder into run history
MODEL_OUTPUT_DIR = './outputs/model'
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

feature_spec = tf.feature_column.make_parse_example_spec(col)
export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)

# Only export predict mode.
# TODO see https://www.tensorflow.org/api_docs/python/tf/contrib/estimator/SavedModelEstimator
# Note, 'export_savedmodel' will be replaced by export_saved_model from tf 2.0
model_path = model.export_savedmodel(MODEL_OUTPUT_DIR, export_input_fn) # TODO strip_default_attrs=True

print("Model saved at", model_path)

run.upload_file()


"""
For details of examples of export and load models,
https://github.com/MtDersvan/tf_playground/blob/master/wide_and_deep_tutorial/wide_and_deep_basic_serving.md
https://www.tensorflow.org/guide/saved_model
https://github.com/monk1337/DNNClassifier-example/
"""
