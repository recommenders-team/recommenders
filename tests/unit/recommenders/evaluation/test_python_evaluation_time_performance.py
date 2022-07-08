# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import minmax_scale

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    SEED,
)
from recommenders.evaluation.python_evaluation import (
    merge_rating_true_pred,
    merge_ranking_true_pred,
    rmse,
    mae,
    rsquared,
    exp_var,
    get_top_k_items,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    auc,
    logloss,
)
import random
from recommenders.utils.timer import Timer

random.seed(SEED)
DATA_USER_NUM = 10000
DATA_ITEM_NUM = DATA_USER_NUM * 2
DATA_SAMPLE_NUM = DATA_USER_NUM * 1000
DATA_RATING_MAX = 5


# fmt: off
@pytest.fixture
def rating_true():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.choice(range(0, DATA_USER_NUM), DATA_SAMPLE_NUM),
            DEFAULT_ITEM_COL: np.random.choice(range(0, DATA_ITEM_NUM), DATA_SAMPLE_NUM),
            DEFAULT_RATING_COL: np.random.choice(range(1, DATA_RATING_MAX+1), DATA_SAMPLE_NUM)
        }
    )


@pytest.fixture
def rating_pred():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.choice(range(0, DATA_USER_NUM), DATA_SAMPLE_NUM),
            DEFAULT_ITEM_COL: np.random.choice(range(0, DATA_ITEM_NUM), DATA_SAMPLE_NUM),
            DEFAULT_PREDICTION_COL: np.random.choice(range(1, DATA_RATING_MAX+1), DATA_SAMPLE_NUM)
        }
    )
# fmt: on


@pytest.fixture
def rating_true_binary(rating_true):
    # Convert true ratings to binary
    rating_true[DEFAULT_RATING_COL] = rating_true[DEFAULT_RATING_COL].apply(
        lambda x: 1.0 if x >= 3 else 0.0
    )
    return rating_true


@pytest.fixture
def rating_pred_binary(rating_pred):
    # Normalize the predictions
    rating_pred[DEFAULT_PREDICTION_COL] = minmax_scale(
        rating_pred[DEFAULT_PREDICTION_COL].astype(float)
    )
    return rating_pred


# The following time thresholds are benchmarked on Azure
# Standard_D14_v2 VM, with Intel(R) Xeon(R) CPU E5-2673 v3 @ 2.40GHz,
# 16 cores, 112 GB RAM, 800 GB disk.
# The thresholds are calculated by MEAN + 5 * STANDARD DEVIATION.


def test_merge_rating(rating_true, rating_pred):
    with Timer() as t:
        merge_rating_true_pred(
            rating_true,
            rating_pred,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 19.58985256


def test_merge_ranking(rating_true, rating_pred):
    with Timer() as t:
        merge_ranking_true_pred(
            rating_true,
            rating_pred,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
            relevancy_method="top_k",
        )
    assert t.interval < 21.52722161


def test_python_rmse(rating_true, rating_pred):
    with Timer() as t:
        rmse(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 29.95031411


def test_python_mae(rating_true, rating_pred):
    with Timer() as t:
        mae(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 30.45756622


def test_python_rsquared(rating_true, rating_pred):
    with Timer() as t:
        rsquared(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 30.60572284


def test_python_exp_var(rating_true, rating_pred):
    with Timer() as t:
        exp_var(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 31.01451915


def test_get_top_k_items(rating_true):
    with Timer() as t:
        get_top_k_items(
            dataframe=rating_true,
            col_user=DEFAULT_USER_COL,
            col_rating=DEFAULT_RATING_COL,
            k=10,
        )
    assert t.interval < 2.94850964


def test_get_top_k_items_largek(rating_true):
    with Timer() as t:
        get_top_k_items(
            dataframe=rating_true,
            col_user=DEFAULT_USER_COL,
            col_rating=DEFAULT_RATING_COL,
            k=1000,
        )
    assert t.interval < 4.04160459


def test_python_ndcg_at_k(rating_true, rating_pred):
    with Timer() as t:
        ndcg_at_k(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
            k=10,
        )
    assert t.interval < 21.55627936


def test_python_map_at_k(rating_true, rating_pred):
    with Timer() as t:
        map_at_k(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
            k=10,
        )
    assert t.interval < 29.90376154


def test_python_precision(rating_true, rating_pred):
    with Timer() as t:
        precision_at_k(rating_true, rating_pred, k=10)
    assert t.interval < 29.95129834


def test_python_recall(rating_true, rating_pred):
    with Timer() as t:
        recall_at_k(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_prediction=DEFAULT_PREDICTION_COL,
            k=10,
        )
    assert t.interval < 30.29558967


def test_python_auc(rating_true_binary, rating_pred_binary):
    with Timer() as t:
        auc(
            rating_true=rating_true_binary,
            rating_pred=rating_pred_binary,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 21.53587225


def test_python_logloss(rating_true_binary, rating_pred_binary):
    with Timer() as t:
        logloss(
            rating_true=rating_true_binary,
            rating_pred=rating_pred_binary,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
        )
    assert t.interval < 32.91629787
