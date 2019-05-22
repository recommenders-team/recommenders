# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys

sys.path.append("../../")

import argparse
import json
import logging
import numpy as np
import os
import pandas as pd

import nni
import surprise

import reco_utils.evaluation.python_evaluation as evaluation
from reco_utils.recommender.surprise.surprise_utils import compute_rating_predictions, compute_ranking_predictions

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("surprise_svd")


def svd_training(params):
    """
    Train Surprise SVD using the given hyper-parameters
    """
    logger.debug("Start training...")
    train_data = pd.read_pickle(path=os.path.join(params['datastore'], params['train_datapath']))
    validation_data = pd.read_pickle(path=os.path.join(params['datastore'], params['validation_datapath']))

    svd_params = {p: params[p] for p in ['random_state', 'n_epochs', 'verbose', 'biased', 'n_factors', 'init_mean',
                                         'init_std_dev', 'lr_all', 'reg_all', 'lr_bu', 'lr_bi', 'lr_pu', 'lr_qi',
                                         'reg_bu', 'reg_bi', 'reg_pu', 'reg_qi']}
    svd = surprise.SVD(**svd_params)

    train_set = surprise.Dataset.load_from_df(train_data, reader=surprise.Reader(params['surprise_reader'])) \
        .build_full_trainset()
    svd.fit(train_set)

    logger.debug("Evaluating...")

    metrics_dict = {}
    rating_metrics = params['rating_metrics']
    if len(rating_metrics) > 0:
        predictions = compute_rating_predictions(svd, validation_data, usercol=params['usercol'],
                                                 itemcol=params['itemcol'])
        for metric in rating_metrics:
            result = getattr(evaluation, metric)(validation_data, predictions)
            logger.debug("%s = %g", metric, result)
            if metric == params['primary_metric']:
                metrics_dict['default'] = result
            else:
                metrics_dict[metric] = result

    ranking_metrics = params['ranking_metrics']
    if len(ranking_metrics) > 0:
        all_predictions = compute_ranking_predictions(svd, train_data, usercol=params['usercol'],
                                                      itemcol=params['itemcol'],
                                                      remove_seen=params['remove_seen'])
        k = params['k']
        for metric in ranking_metrics:
            result = getattr(evaluation, metric)(validation_data, all_predictions, col_prediction='prediction', k=k)
            logger.debug("%s@%d = %g", metric, k, result)
            if metric == params['primary_metric']:
                metrics_dict['default'] = result
            else:
                metrics_dict[metric] = result

    if len(ranking_metrics) == 0 and len(rating_metrics) == 0:
        raise ValueError("No metrics were specified.")

    # Report the metrics
    nni.report_final_result(metrics_dict)

    # Save the metrics in a JSON file
    output_dir = os.environ.get('NNI_OUTPUT_DIR')
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as fp:
        temp_dict = metrics_dict.copy()
        temp_dict[params['primary_metric']] = temp_dict.pop('default')
        json.dump(temp_dict, fp)

    return svd


def get_params():
    parser = argparse.ArgumentParser()
    # Data path
    parser.add_argument('--datastore', type=str, dest='datastore', help="Datastore path")
    parser.add_argument('--train-datapath', type=str, dest='train_datapath')
    parser.add_argument('--validation-datapath', type=str, dest='validation_datapath')
    parser.add_argument('--surprise-reader', type=str, dest='surprise_reader')
    parser.add_argument('--usercol', type=str, dest='usercol', default='userID')
    parser.add_argument('--itemcol', type=str, dest='itemcol', default='itemID')
    # Metrics
    parser.add_argument('--rating-metrics', type=str, nargs='*', dest='rating_metrics', default=[])
    parser.add_argument('--ranking-metrics', type=str, nargs='*', dest='ranking_metrics', default=[])
    parser.add_argument('--k', type=int, dest='k', default=None)
    parser.add_argument('--remove-seen', dest='remove_seen', action='store_false')
    # Training parameters
    parser.add_argument('--random-state', type=int, dest='random_state', default=0)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--epochs', type=int, dest='n_epochs', default=30)
    parser.add_argument('--biased', dest='biased', action='store_true')
    parser.add_argument('--primary-metric', dest='primary_metric', default='rmse')
    # Hyperparameters to be tuned
    parser.add_argument('--n_factors', type=int, dest='n_factors', default=100)
    parser.add_argument('--init_mean', type=float, dest='init_mean', default=0.0)
    parser.add_argument('--init_std_dev', type=float, dest='init_std_dev', default=0.1)
    parser.add_argument('--lr_all', type=float, dest='lr_all', default=0.005)
    parser.add_argument('--reg_all', type=float, dest='reg_all', default=0.02)
    parser.add_argument('--lr_bu', type=float, dest='lr_bu', default=None)
    parser.add_argument('--lr_bi', type=float, dest='lr_bi', default=None)
    parser.add_argument('--lr_pu', type=float, dest='lr_pu', default=None)
    parser.add_argument('--lr_qi', type=float, dest='lr_qi', default=None)
    parser.add_argument('--reg_bu', type=float, dest='reg_bu', default=None)
    parser.add_argument('--reg_bi', type=float, dest='reg_bi', default=None)
    parser.add_argument('--reg_pu', type=float, dest='reg_pu', default=None)
    parser.add_argument('--reg_qi', type=float, dest='reg_qi', default=None)

    args = parser.parse_args()
    return args


def main(params):
    logger.debug("Args: %s", str(params))
    logger.debug('Number of epochs %d', params["n_epochs"])

    svd = svd_training(params)
    # Save SVD model to the output directory for later use
    output_dir = os.environ.get('NNI_OUTPUT_DIR')
    surprise.dump.dump(os.path.join(output_dir, 'model.dump'), algo=svd)


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        logger.debug("Hyperparameters: %s", tuner_params)
        params = vars(get_params())
        # in the case of Hyperband, use STEPS to allocate the number of epochs SVD will run for 
        if 'STEPS' in tuner_params:
            steps_param = tuner_params['STEPS']
            params['n_epochs'] = int(np.rint(steps_param))
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
