# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
AzureML Hyperdrive entry script for wide-deep model
"""
import argparse
import os
import shutil

import papermill as pm
import tensorflow as tf

print("TensorFlow version:", tf.VERSION)

try:
    from azureml.core import Run

    run = Run.get_context()
except ImportError:
    run = None

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)


NOTEBOOK_NAME = os.path.join("notebooks", "00_quick_start", "wide_deep_movielens.ipynb")
OUTPUT_NOTEBOOK = "wide_deep.ipynb"


def _log(metric, value):
    """AzureML log wrapper.

    Record list of int or float as a list metrics so that we can plot it from AzureML workspace portal.
    Otherwise, record as a single value of the metric.
    """
    if run is not None:
        if (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], (int, float))
        ):
            run.log_list(metric, value)
        else:
            # Force cast to str since run.log will raise an error if the value is iterable.
            run.log(metric, str(value))
    print(metric, "=", value)


# Parse arguments passed by Hyperdrive
parser = argparse.ArgumentParser()

parser.add_argument(
    "--top-k", type=int, dest="TOP_K", help="Top k recommendation", default=10
)
# Data path
parser.add_argument("--datastore", type=str, dest="DATA_DIR", help="Datastore path")
parser.add_argument("--train-datapath", type=str, dest="TRAIN_PICKLE_PATH")
parser.add_argument("--test-datapath", type=str, dest="TEST_PICKLE_PATH")
parser.add_argument(
    "--model-dir", type=str, dest="MODEL_DIR", default="model_checkpoints"
)
# Data column names
parser.add_argument("--user-col", type=str, dest="USER_COL", default=DEFAULT_USER_COL)
parser.add_argument("--item-col", type=str, dest="ITEM_COL", default=DEFAULT_ITEM_COL)
parser.add_argument(
    "--rating-col", type=str, dest="RATING_COL", default=DEFAULT_RATING_COL
)
parser.add_argument("--item-feat-col", type=str, dest="ITEM_FEAT_COL")  # Optional
parser.add_argument(
    "--ranking-metrics",
    type=str,
    nargs="*",
    dest="RANKING_METRICS",
    default=["ndcg_at_k"],
)
parser.add_argument(
    "--rating-metrics", type=str, nargs="*", dest="RATING_METRICS", default=["rmse"]
)
# Model type: either 'wide', 'deep', or 'wide_deep'
parser.add_argument("--model-type", type=str, dest="MODEL_TYPE", default="wide_deep")
# Wide model params
parser.add_argument(
    "--linear-optimizer", type=str, dest="LINEAR_OPTIMIZER", default="Ftrl"
)
parser.add_argument(
    "--linear-optimizer-lr", type=float, dest="LINEAR_OPTIMIZER_LR", default=0.01
)
parser.add_argument("--linear-l1-reg", type=float, dest="LINEAR_L1_REG", default=0.0)
parser.add_argument("--linear-l2-reg", type=float, dest="LINEAR_L2_REG", default=0.0)
parser.add_argument(
    "--linear-momentum", type=float, dest="LINEAR_MOMENTUM", default=0.9
)
# Deep model params
parser.add_argument(
    "--dnn-optimizer", type=str, dest="DNN_OPTIMIZER", default="Adagrad"
)
parser.add_argument(
    "--dnn-optimizer-lr", type=float, dest="DNN_OPTIMIZER_LR", default=0.01
)
parser.add_argument("--dnn-l1-reg", type=float, dest="DNN_L1_REG", default=0.0)
parser.add_argument("--dnn-l2-reg", type=float, dest="DNN_L2_REG", default=0.0)
parser.add_argument("--dnn-momentum", type=float, dest="DNN_MOMENTUM", default=0.9)
parser.add_argument(
    "--dnn-hidden-layer-1", type=int, dest="DNN_HIDDEN_LAYER_1", default=0
)
parser.add_argument(
    "--dnn-hidden-layer-2", type=int, dest="DNN_HIDDEN_LAYER_2", default=0
)
parser.add_argument(
    "--dnn-hidden-layer-3", type=int, dest="DNN_HIDDEN_LAYER_3", default=128
)
parser.add_argument(
    "--dnn-hidden-layer-4", type=int, dest="DNN_HIDDEN_LAYER_4", default=128
)
parser.add_argument(
    "--dnn-user-embedding-dim", type=int, dest="DNN_USER_DIM", default=8
)
parser.add_argument(
    "--dnn-item-embedding-dim", type=int, dest="DNN_ITEM_DIM", default=8
)
parser.add_argument("--dnn-batch-norm", type=int, dest="DNN_BATCH_NORM", default=1)
parser.add_argument("--dnn-dropout", type=float, dest="DNN_DROPOUT", default=0.0)
# Training parameters
parser.add_argument("--steps", type=int, dest="STEPS", default=10000)
parser.add_argument("--batch-size", type=int, dest="BATCH_SIZE", default=128)
parser.add_argument(
    "--evaluate-while-training", dest="EVALUATE_WHILE_TRAINING", action="store_true"
)


args = parser.parse_args()

params = vars(args)

if params["TOP_K"] <= 0:
    raise ValueError("Top K should be larger than 0")

if params["MODEL_TYPE"] not in {"wide", "deep", "wide_deep"}:
    raise ValueError("Model type should be either 'wide', 'deep', or 'wide_deep'")

if params["DATA_DIR"] is None:
    raise ValueError("Datastore path should be given")

print("Args:")
for k, v in params.items():
    _log(k, v)

print("Run", NOTEBOOK_NAME)

pm.execute_notebook(
    NOTEBOOK_NAME, OUTPUT_NOTEBOOK, parameters=params, kernel_name="python3"
)
nb = pm.read_notebook(OUTPUT_NOTEBOOK)

for m, v in nb.data.items():
    _log(m, v)

# clean-up
os.remove(OUTPUT_NOTEBOOK)
shutil.rmtree(params["MODEL_DIR"], ignore_errors=True)
