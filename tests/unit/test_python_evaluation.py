# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from mock import Mock
from sklearn.preprocessing import minmax_scale
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    PREDICTION_COL
)
from reco_utils.evaluation.python_evaluation import (
    check_column_dtypes,
    merge_rating_true_pred,
    merge_ranking_true_pred,
    rmse,
    mae,
    rsquared,
    exp_var,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    auc,
    logloss
)

TOL = 0.0001


@pytest.fixture(scope="module")
def target_metrics():
    return {
        "rmse": pytest.approx(7.254309, TOL),
        "mae": pytest.approx(6.375, TOL),
        "rsquared": pytest.approx(-31.699029, TOL),
        "exp_var": pytest.approx(-6.4466, 0.01),
        "ndcg": pytest.approx(0.38172, TOL),
        "precision": pytest.approx(0.26666, TOL),
        "map": pytest.approx(0.23613, TOL),
        "recall": pytest.approx(0.37777, TOL),
        "auc": pytest.approx(0.75, TOL),
        "logloss": pytest.approx(0.7835, TOL)
    }


@pytest.fixture(scope="module")
def python_data():
    def _generate_python_data(binary_rating=False):
        rating_true = pd.DataFrame(
            {
                DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                DEFAULT_ITEM_COL: [1, 2, 3, 1, 4, 5, 6, 7, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
                DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1],
            }
        )
        rating_pred = pd.DataFrame(
            {
                DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                DEFAULT_ITEM_COL: [3, 10, 12, 10, 3, 5, 11, 13, 4, 10, 7, 13, 1, 3, 5, 2, 11, 14],
                PREDICTION_COL: [
                    14,
                    13,
                    12,
                    14,
                    13,
                    12,
                    11,
                    10,
                    14,
                    13,
                    12,
                    11,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                ],
            }
        )
        rating_nohit = pd.DataFrame(
            {
                DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                DEFAULT_ITEM_COL: [100] * rating_pred.shape[0],
                PREDICTION_COL: [
                    14,
                    13,
                    12,
                    14,
                    13,
                    12,
                    11,
                    10,
                    14,
                    13,
                    12,
                    11,
                    10,
                    9,
                    8,
                    7,
                    6,
                    5,
                ],
            }
        )

        if binary_rating:
            # Convert to binary case.
            rating_true[DEFAULT_RATING_COL] = rating_true[DEFAULT_RATING_COL].apply(lambda x: 1.0 if x >= 3 else 0.0)

            # Normalize the prediction.
            rating_pred[PREDICTION_COL] = minmax_scale(rating_pred[PREDICTION_COL].astype(float))

        return rating_true, rating_pred, rating_nohit
    return _generate_python_data


def test_column_dtypes_match(python_data):
    rating_true, rating_pred, _ = python_data(binary_rating=False)

    # Change data types of true and prediction data, and there should type error produced.
    rating_true_copy = rating_true.copy()

    rating_true_copy[DEFAULT_USER_COL] = rating_true_copy[DEFAULT_USER_COL].astype(str)
    rating_true_copy[DEFAULT_RATING_COL] = rating_true_copy[DEFAULT_RATING_COL].astype(str)

    with pytest.raises(TypeError) as e_info:
        f = Mock()
        f_d = check_column_dtypes(f)
        f_d(
            rating_true_copy,
            rating_pred,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=PREDICTION_COL
        )

        # Error message is expected when there is mismatch.
        assert str(e_info.value) == "Data types of column {} are different in true and prediction".format(DEFAULT_USER_COL)


def test_merge_rating(python_data):
    rating_true, rating_pred, _ = python_data(binary_rating=False)

    y_true, y_pred = merge_rating_true_pred(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL
    )
    
    assert isinstance(y_true, np.ndarray)
    assert isinstance(y_pred, np.ndarray)
    assert y_true.shape == y_pred.shape


def test_merge_ranking(python_data):
    ranking_true, ranking_pred, _ = python_data(binary_rating=False)

    ranking_true_pred, data_hit, n_users = merge_ranking_true_pred(
        ranking_true,
        ranking_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL,
        relevancy_method="top_k"
    )

    assert isinstance(ranking_true_pred, pd.DataFrame)

    columns = ranking_true_pred.columns
    columns_exp = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, PREDICTION_COL]
    assert set(columns).intersection(set(columns_exp)) is not None

    assert isinstance(data_hit, pd.DataFrame)

    assert isinstance(n_users, int)


def test_python_rmse(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data(binary_rating=False)
    assert rmse(rating_true=rating_true, rating_pred=rating_true, col_prediction=DEFAULT_RATING_COL) == 0
    assert rmse(rating_true, rating_pred) == target_metrics["rmse"]


def test_python_mae(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data(binary_rating=False)
    assert mae(rating_true=rating_true, rating_pred=rating_true, col_prediction=DEFAULT_RATING_COL) == 0
    assert mae(rating_true, rating_pred) == target_metrics["mae"]


def test_python_rsquared(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data(binary_rating=False)

    assert rsquared(
        rating_true=rating_true, rating_pred=rating_true, col_prediction=DEFAULT_RATING_COL
    ) == pytest.approx(1.0, TOL)
    assert rsquared(rating_true, rating_pred) == target_metrics["rsquared"]


def test_python_exp_var(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data(binary_rating=False)

    assert exp_var(
        rating_true=rating_true, rating_pred=rating_true, col_prediction=DEFAULT_RATING_COL
    ) == pytest.approx(1.0, TOL)
    assert exp_var(rating_true, rating_pred) == target_metrics["exp_var"]


def test_python_ndcg_at_k(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data(binary_rating=False)

    assert ndcg_at_k(
        k=10,
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
    ) == 1
    assert ndcg_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert ndcg_at_k(rating_true, rating_pred, k=10) == target_metrics["ndcg"]


def test_python_map_at_k(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data(binary_rating=False)

    assert map_at_k(
        k=10,
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
    ) == 1
    assert map_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert map_at_k(rating_true, rating_pred, k=10) == target_metrics["map"]


def test_python_precision(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data(binary_rating=False)
    assert precision_at_k(
        k=10,
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
    ) == 0.6
    assert precision_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert precision_at_k(rating_true, rating_pred, k=10) == target_metrics["precision"]

    # Check normalization
    single_user = pd.DataFrame(
        {DEFAULT_USER_COL: [1, 1, 1], DEFAULT_ITEM_COL: [1, 2, 3], DEFAULT_RATING_COL: [5, 4, 3]}
    )
    assert precision_at_k(
        k=3,
        rating_true=single_user,
        rating_pred=single_user,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_RATING_COL,
    ) == 1

    same_items = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2],
            DEFAULT_ITEM_COL: [1, 2, 3, 1, 2, 3],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3],
        }
    )
    assert precision_at_k(
        k=3,
        rating_true=same_items,
        rating_pred=same_items,
        col_prediction=DEFAULT_RATING_COL
    ) == 1

    # Check that if the sample size is smaller than k, the maximum precision can not be 1
    # if we do precision@5 when there is only 3 items, we can get a maximum of 3/5.
    assert precision_at_k(
        k=5, rating_true=same_items, rating_pred=same_items, col_prediction=DEFAULT_RATING_COL
    ) == 0.6


def test_python_recall(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data(binary_rating=False)

    assert recall_at_k(
        k=10, rating_true=rating_true, rating_pred=rating_true, col_prediction=DEFAULT_RATING_COL
    ) == pytest.approx(1, TOL)
    assert recall_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert recall_at_k(rating_true, rating_pred, k=10) == target_metrics["recall"]


def test_python_auc(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data(binary_rating=True)

    assert auc(
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL
    ) == pytest.approx(1.0, TOL)

    assert auc(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL
    ) == target_metrics['auc']


def test_python_logloss(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data(binary_rating=True)

    assert logloss(
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL
    ) == pytest.approx(0, TOL)

    assert logloss(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL
    ) == target_metrics['logloss']


def test_python_errors(python_data):
    rating_true, rating_pred, _ = python_data(binary_rating=False)

    with pytest.raises(ValueError):
        rmse(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        mae(rating_pred, rating_pred, col_rating=PREDICTION_COL, col_user="not_user")

    with pytest.raises(ValueError):
        rsquared(rating_true, rating_pred, col_item="not_item")

    with pytest.raises(ValueError):
        exp_var(rating_pred, rating_pred, col_rating=PREDICTION_COL, col_item="not_item")

    with pytest.raises(ValueError):
        precision_at_k(rating_true, rating_pred, col_rating="not_rating")

    with pytest.raises(ValueError):
        recall_at_k(rating_true, rating_pred, col_prediction="not_prediction")

    with pytest.raises(ValueError):
        ndcg_at_k(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        map_at_k(rating_pred, rating_pred, col_rating=PREDICTION_COL, col_user="not_user")
