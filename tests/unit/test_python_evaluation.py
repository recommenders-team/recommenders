# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
import pytest

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    PREDICTION_COL
)
from reco_utils.evaluation.python_evaluation import RatingEvaluation, RankingEvaluation


TOL = 0.0001


@pytest.fixture(scope='module')
def df_true():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5, 6, 7, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
        DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1],
    })


@pytest.fixture(scope='module')
def df_pred():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        DEFAULT_ITEM_COL: [3, 10, 12, 10, 3, 5, 11, 13, 4, 10, 7, 13, 1, 3, 5, 2, 11, 14],
        PREDICTION_COL: [14, 13, 12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
    })


@pytest.fixture(scope='module')
def df_true_binary(df_true):
    df_true_binary = df_true.copy()
    df_true_binary[DEFAULT_RATING_COL] = df_true[DEFAULT_RATING_COL].apply(lambda x: 1.0 if x >= 3 else 0.0)
    return df_true_binary


@pytest.fixture(scope='module')
def df_pred_binary(df_pred):
    df_pred_binary = df_pred.copy()
    df_pred_binary[PREDICTION_COL] = minmax_scale(df_pred[PREDICTION_COL].astype('float64'))
    return df_pred_binary


@pytest.fixture(scope='module')
def df_nohit():
    return pd.DataFrame({
        DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        DEFAULT_ITEM_COL: [100] * 18,
        PREDICTION_COL: [14, 13, 12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
    })


def test_rating_evaluation(df_true, df_pred):
    evaluator = RatingEvaluation(df_true=df_true, df_pred=df_pred)
    assert evaluator.exp_var == pytest.approx(-6.4466, TOL)
    assert evaluator.mae == pytest.approx(6.375, TOL)
    assert evaluator.rmse == pytest.approx(7.254309, TOL)
    assert evaluator.rsquared == pytest.approx(-31.699029, TOL)


def test_binary_rating_evaluation(df_true_binary, df_pred_binary):
    evaluator = RatingEvaluation(df_true=df_true_binary, df_pred=df_pred_binary)
    assert evaluator.auc == pytest.approx(0.75, TOL)
    assert evaluator.logloss == pytest.approx(0.7835, TOL)


def test_perfect_rating_evaluation(df_true):
    evaluator = RatingEvaluation(df_true=df_true, df_pred=df_true, col_prediction=DEFAULT_RATING_COL)
    assert evaluator.exp_var == 1.0
    assert evaluator.mae == 0.0
    assert evaluator.rmse == 0.0
    assert evaluator.rsquared == 1.0


def test_perfect_binary_rating_evaluation(df_true_binary):
    evaluator = RatingEvaluation(df_true=df_true_binary, df_pred=df_true_binary, col_prediction=DEFAULT_RATING_COL)
    assert evaluator.auc == 1.0
    assert evaluator.logloss == pytest.approx(0.0, TOL)


def test_nohit_rating_evaluation(df_true, df_nohit):
    evaluator = RatingEvaluation(df_true=df_true, df_pred=df_nohit)
    assert evaluator.exp_var == 0.0
    assert np.isinf(evaluator.mae)
    assert np.isinf(evaluator.rmse)
    assert evaluator.rsquared == 0.0
    assert evaluator.auc == 0.0
    assert np.isinf(evaluator.logloss)


def test_ranking_evaluation(df_true, df_pred):
    evaluator = RankingEvaluation(df_true=df_true, df_pred=df_pred)
    assert evaluator.ndcg_at_k == pytest.approx(0.38172, TOL)
    assert evaluator.map_at_k == pytest.approx(0.23613, TOL)
    assert evaluator.precision_at_k == pytest.approx(0.26666, TOL)
    assert evaluator.recall_at_k == pytest.approx(0.37777, TOL)


def test_nohit_ranking_evaluation(df_true, df_nohit):
    evaluator = RankingEvaluation(df_true=df_true, df_pred=df_nohit)
    assert evaluator.ndcg_at_k == 0.0
    assert evaluator.map_at_k == 0.0
    assert evaluator.precision_at_k == 0.0
    assert evaluator.recall_at_k == 0.0
