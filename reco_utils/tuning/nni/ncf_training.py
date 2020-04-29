# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
import nni
import sys

sys.path.append("../../../")

import reco_utils.evaluation.python_evaluation as evaluation
from reco_utils.recommender.ncf.ncf_singlenode import NCF
from reco_utils.recommender.ncf.dataset import Dataset as NCFDataset
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_chrono_split
from reco_utils.evaluation.python_evaluation import (rmse, mae, rsquared, exp_var, map_at_k, ndcg_at_k, precision_at_k, 
                                                     recall_at_k, get_top_k_items)
from reco_utils.common.constants import SEED as DEFAULT_SEED

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ncf")


def ncf_training(params):
    """
    Train NCF using the given hyper-parameters
    """
    logger.debug("Start training...")
    train_data = pd.read_pickle(
        path=os.path.join(params["datastore"], params["train_datapath"])
    )
    validation_data = pd.read_pickle(
        path=os.path.join(params["datastore"], params["validation_datapath"])
    )

    data = NCFDataset(train=train_data, test=validation_data, seed=DEFAULT_SEED)

    model = NCF (
        n_users=data.n_users, 
        n_items=data.n_items,
        model_type="NeuMF",
        n_factors=params["n_factors"],
        layer_sizes=[16,8,4],
        n_epochs=params["n_epochs"],
        learning_rate=params["learning_rate"],
        verbose=params["verbose"],
        seed=DEFAULT_SEED
    )

    model.fit(data)

    logger.debug("Evaluating...")

    metrics_dict = {}
    rating_metrics = params["rating_metrics"]
    if len(rating_metrics) > 0:
        predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
               for (_, row) in validation_data.iterrows()]

        predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
        predictions = predictions.astype({'userID': 'int64', 'itemID': 'int64', 'prediction': 'float64'})
        
        for metric in rating_metrics:
            result = getattr(evaluation, metric)(validation_data, predictions)
            logger.debug("%s = %g", metric, result)
            if metric == params["primary_metric"]:
                metrics_dict["default"] = result
            else:
                metrics_dict[metric] = result

    ranking_metrics = params["ranking_metrics"]
    if len(ranking_metrics) > 0:
        users, items, preds = [], [], []
        item = list(train_data.itemID.unique())
        for user in train_data.userID.unique():
            user = [user] * len(item) 
            users.extend(user)
            items.extend(item)
            preds.extend(list(model.predict(user, item, is_list=True)))

        all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})

        merged = pd.merge(train_data, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)
        for metric in ranking_metrics:
            result = getattr(evaluation, metric)(
                validation_data, all_predictions, col_prediction="prediction", k=params["k"]
            )
            logger.debug("%s@%d = %g", metric, params["k"], result)
            if metric == params["primary_metric"]:
                metrics_dict["default"] = result
            else:
                metrics_dict[metric] = result

    if len(ranking_metrics) == 0 and len(rating_metrics) == 0:
        raise ValueError("No metrics were specified.")

    # Report the metrics
    nni.report_final_result(metrics_dict)

    # Save the metrics in a JSON file
    output_dir = os.environ.get("NNI_OUTPUT_DIR")
    with open(os.path.join(output_dir, "metrics.json"), "w") as fp:
        temp_dict = metrics_dict.copy()
        temp_dict[params["primary_metric"]] = temp_dict.pop("default")
        json.dump(temp_dict, fp)

    return model


def get_params():
    parser = argparse.ArgumentParser()
    # Data path
    parser.add_argument(
        "--datastore", type=str, dest="datastore", help="Datastore path"
    )
    parser.add_argument("--train-datapath", type=str, dest="train_datapath")
    parser.add_argument("--validation-datapath", type=str, dest="validation_datapath")
    parser.add_argument("--surprise-reader", type=str, dest="surprise_reader")
    parser.add_argument("--usercol", type=str, dest="usercol", default="userID")
    parser.add_argument("--itemcol", type=str, dest="itemcol", default="itemID")
    # Metrics
    parser.add_argument(
        "--rating-metrics", type=str, nargs="*", dest="rating_metrics", default=[]
    )
    parser.add_argument(
        "--ranking-metrics", type=str, nargs="*", dest="ranking_metrics", default=[]
    )
    parser.add_argument("--k", type=int, dest="k", default=None)
    parser.add_argument("--remove-seen", dest="remove_seen", action="store_false")
    # Training parameters
    parser.add_argument("--random-state", type=int, dest="random_state", default=0)
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.add_argument("--epochs", type=int, dest="n_epochs", default=30)
    parser.add_argument("--biased", dest="biased", action="store_true")
    parser.add_argument("--primary-metric", dest="primary_metric", default="rmse")
    # Hyperparameters to be tuned
    parser.add_argument("--n_factors", type=int, dest="n_factors", default=100)

    args = parser.parse_args()
    return args


def main(params):
    logger.debug("Args: %s", str(params))
    logger.debug("Number of epochs %d", params["n_epochs"])

    model = ncf_training(params)
    # Save NCF model to the output directory for later use
    output_dir = os.environ.get("NNI_OUTPUT_DIR")
    # surprise.dump.dump(os.path.join(output_dir, "model.dump"), algo=svd)


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug("Hyperparameters: %s", tuner_params)
        params = vars(get_params())
        # in the case of Hyperband, use STEPS to allocate the number of epochs SVD will run for
        if "STEPS" in tuner_params:
            steps_param = tuner_params["STEPS"]
            params["n_epochs"] = int(np.rint(steps_param))
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
