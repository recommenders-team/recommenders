# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys

sys.path.append("../../")

import argparse
import os
import pandas as pd
import surprise

try:
    from azureml.core import Run
    HAS_AML = True
except ModuleNotFoundError:
    HAS_AML = False

from reco_utils.evaluation.python_evaluation import (
    rmse, mae, rsquared, exp_var,
    map_at_k, ndcg_at_k, precision_at_k, recall_at_k
)

parser = argparse.ArgumentParser()
# Data path
parser.add_argument('--datastore', type=str, dest='datastore', help="Datastore path")
parser.add_argument('--train-datapath', type=str, dest='train_datapath')
parser.add_argument('--test-datapath', type=str, dest='test_datapath')
parser.add_argument('--surprise-reader', type=str, dest='surprise_reader')
# Metrics
parser.add_argument('--metrics', type=str, nargs='*', dest='metrics', default=['rmse'])
# Training parameters
parser.add_argument('--random-state', type=int, dest='random_state', default=0)
parser.add_argument('--verbose', dest='verbose', action='store_true')
parser.add_argument('--epochs', type=int, dest='epochs', default=30)
parser.add_argument('--biased', dest='biased', action='store_true')
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

RANDOM_STATE = args.random_state
VERBOSE = args.verbose
NUM_EPOCHS = args.epochs
BIASED = args.biased

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

LOG_STEPS = 1
print("Args:", str(vars(args)), sep='\n')

# Get AML run context
if HAS_AML:
    run = Run.get_context()
    run.log('Number of epochs', NUM_EPOCHS)


def svd_training(train_data, test_data):
    """
    Train Surprise SVD using the given hyper-parameters
    """

    print("Start training...")
    svd = surprise.SVD(random_state=RANDOM_STATE, n_epochs=NUM_EPOCHS, verbose=VERBOSE, biased=BIASED,
                       n_factors=args.n_factors, init_mean=args.init_mean, init_std_dev=args.init_std_dev,
                       lr_all=args.lr_all, reg_all=args.reg_all, lr_bu=args.lr_bu, lr_bi=args.lr_bi, lr_pu=args.lr_pu,
                       lr_qi=args.lr_qi, reg_bu=args.reg_bu, reg_bi=args.reg_bi, reg_pu=args.reg_pu,
                       reg_qi=args.reg_qi)

    train_set = surprise.Dataset.load_from_df(train_data, reader=surprise.Reader(args.surprise_reader)) \
        .build_full_trainset()

    svd.fit(train_set)

    print("Evaluating...")

    if len(RATING_METRICS) > 0:
        predictions = [svd.predict(row.userID, row.itemID, row.rating)
                       for (_, row) in test_data.iterrows()]
        predictions = pd.DataFrame(predictions)
        predictions = predictions.rename(index=str, columns={'uid': 'userID', 'iid': 'itemID', 'est': 'prediction'})
        predictions = predictions.drop(['details', 'r_ui'], axis='columns')

        for metric in RATING_METRICS:
            result = _SUPPORTED_METRICS[metric](test_data, predictions)
            print(metric, result)
            if HAS_AML:
                run.log(metric, result)

    if len(RANKING_METRICS) > 0:
        preds_lst = []
        for user in train_data.userID.unique():
            for item in train_data.itemID.unique():
                preds_lst.append([user, item, svd.predict(user, item).est])

        all_predictions = pd.DataFrame(data=preds_lst, columns=["userID", "itemID", "prediction"])

        merged = pd.merge(train_data, all_predictions, on=["userID", "itemID"], how="outer")
        all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

        for metric in RANKING_METRICS:
            str_k = metric.split('@')
            result = _SUPPORTED_METRICS[str_k[0]](test_data, all_predictions, col_prediction='prediction',
                                                  k=int(str_k[1]))
            print(metric, result)
            if HAS_AML:
                run.log(metric, result)

svd_training(train_data=pd.read_pickle(path=os.path.join(args.datastore, args.train_datapath)),
             test_data=pd.read_pickle(path=os.path.join(args.datastore, args.test_datapath)))
