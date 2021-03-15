# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock
from sklearn.preprocessing import minmax_scale
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
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
    logloss,
)

TOL = 0.0001


@pytest.fixture
def rating_true():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1,],
            DEFAULT_ITEM_COL: [
                3,
                1,
                4,
                5,
                6,
                7,
                2,
                5,
                6,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                1,
                2,
            ],
            DEFAULT_RATING_COL: [3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1, 5, 4,],
        }
    )


@pytest.fixture
def rating_pred():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1,],
            DEFAULT_ITEM_COL: [
                12,
                10,
                3,
                5,
                11,
                13,
                4,
                10,
                7,
                13,
                1,
                3,
                5,
                2,
                11,
                14,
                3,
                10,
            ],
            DEFAULT_PREDICTION_COL: [
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
                14,
                13,
            ],
            DEFAULT_RATING_COL: [3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1, 5, 4,],
        }
    )


@pytest.fixture
def rating_nohit():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1,],
            DEFAULT_ITEM_COL: [100] * 18,
            DEFAULT_PREDICTION_COL: [
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
                14,
                13,
            ],
        }
    )


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


def test_column_dtypes_match(rating_true, rating_pred):
    # Change data types of true and prediction data, and there should type error produced
    rating_true[DEFAULT_USER_COL] = rating_true[DEFAULT_USER_COL].astype(str)
    rating_true[DEFAULT_RATING_COL] = rating_true[DEFAULT_RATING_COL].astype(str)

    expected_error = "Columns in provided DataFrames are not the same datatype"
    with pytest.raises(ValueError, match=expected_error):
        check_column_dtypes(Mock())(
            rating_true,
            rating_pred,
            col_user=DEFAULT_USER_COL,
            col_item=DEFAULT_ITEM_COL,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_PREDICTION_COL,
        )


def test_merge_rating(rating_true, rating_pred):
    y_true, y_pred = merge_rating_true_pred(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    )
    target_y_true = np.array([3, 3, 5, 5, 3, 3, 2, 1])
    target_y_pred = np.array([14, 12, 7, 8, 13, 6, 11, 5])

    assert y_true.shape == y_pred.shape
    assert np.all(y_true == target_y_true)
    assert np.all(y_pred == target_y_pred)


def test_merge_ranking(rating_true, rating_pred):

    data_hit, data_hit_count, n_users = merge_ranking_true_pred(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        relevancy_method="top_k",
    )

    assert isinstance(data_hit, pd.DataFrame)

    assert isinstance(data_hit_count, pd.DataFrame)
    columns = data_hit_count.columns
    columns_exp = [DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL]
    assert set(columns).intersection(set(columns_exp)) is not None

    assert n_users == 3


def test_python_rmse(rating_true, rating_pred):
    assert (
        rmse(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
        )
        == 0
    )
    assert rmse(rating_true, rating_pred) == pytest.approx(7.254309, TOL)


def test_python_mae(rating_true, rating_pred):
    assert (
        mae(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
        )
        == 0
    )
    assert mae(rating_true, rating_pred) == pytest.approx(6.375, TOL)


def test_python_rsquared(rating_true, rating_pred):
    assert rsquared(
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
    ) == pytest.approx(1.0, TOL)
    assert rsquared(rating_true, rating_pred) == pytest.approx(-31.699029, TOL)


def test_python_exp_var(rating_true, rating_pred):
    assert exp_var(
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
    ) == pytest.approx(1.0, TOL)
    assert exp_var(rating_true, rating_pred) == pytest.approx(-6.4466, TOL)


def test_python_ndcg_at_k(rating_true, rating_pred, rating_nohit):
    assert (
        ndcg_at_k(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
            k=10,
        )
        == 1
    )
    assert ndcg_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert ndcg_at_k(rating_true, rating_pred, k=10) == pytest.approx(0.38172, TOL)


def test_python_map_at_k(rating_true, rating_pred, rating_nohit):
    assert (
        map_at_k(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
            k=10,
        )
        == 1
    )
    assert map_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert map_at_k(rating_true, rating_pred, k=10) == pytest.approx(0.23613, TOL)


def test_python_precision(rating_true, rating_pred, rating_nohit):
    assert (
        precision_at_k(
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction=DEFAULT_RATING_COL,
            k=10,
        )
        == 0.6
    )
    assert precision_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert precision_at_k(rating_true, rating_pred, k=10) == pytest.approx(0.26666, TOL)

    # Check normalization
    single_user = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1],
            DEFAULT_ITEM_COL: [1, 2, 3],
            DEFAULT_RATING_COL: [5, 4, 3],
        }
    )
    assert (
        precision_at_k(
            rating_true=single_user,
            rating_pred=single_user,
            col_rating=DEFAULT_RATING_COL,
            col_prediction=DEFAULT_RATING_COL,
            k=3,
        )
        == 1
    )

    same_items = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2],
            DEFAULT_ITEM_COL: [1, 2, 3, 1, 2, 3],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3],
        }
    )
    assert (
        precision_at_k(
            rating_true=same_items,
            rating_pred=same_items,
            col_prediction=DEFAULT_RATING_COL,
            k=3,
        )
        == 1
    )

    # Check that if the sample size is smaller than k, the maximum precision can not be 1
    # if we do precision@5 when there is only 3 items, we can get a maximum of 3/5.
    assert (
        precision_at_k(
            rating_true=same_items,
            rating_pred=same_items,
            col_prediction=DEFAULT_RATING_COL,
            k=5,
        )
        == 0.6
    )


def test_python_recall(rating_true, rating_pred, rating_nohit):
    assert recall_at_k(
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
        k=10,
    ) == pytest.approx(1, TOL)
    assert recall_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert recall_at_k(rating_true, rating_pred, k=10) == pytest.approx(0.37777, TOL)


def test_python_auc(rating_true_binary, rating_pred_binary):
    assert auc(
        rating_true=rating_true_binary,
        rating_pred=rating_true_binary,
        col_prediction=DEFAULT_RATING_COL,
    ) == pytest.approx(1.0, TOL)

    assert auc(
        rating_true=rating_true_binary,
        rating_pred=rating_pred_binary,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    ) == pytest.approx(0.75, TOL)


def test_python_logloss(rating_true_binary, rating_pred_binary):
    assert logloss(
        rating_true=rating_true_binary,
        rating_pred=rating_true_binary,
        col_prediction=DEFAULT_RATING_COL,
    ) == pytest.approx(0, TOL)

    assert logloss(
        rating_true=rating_true_binary,
        rating_pred=rating_pred_binary,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    ) == pytest.approx(0.7835, TOL)


def test_python_errors(rating_true, rating_pred):
    with pytest.raises(ValueError):
        rmse(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        mae(
            rating_pred,
            rating_pred,
            col_rating=DEFAULT_PREDICTION_COL,
            col_user="not_user",
        )

    with pytest.raises(ValueError):
        rsquared(rating_true, rating_pred, col_item="not_item")

    with pytest.raises(ValueError):
        exp_var(
            rating_pred,
            rating_pred,
            col_rating=DEFAULT_PREDICTION_COL,
            col_item="not_item",
        )

    with pytest.raises(ValueError):
        precision_at_k(rating_true, rating_pred, col_rating="not_rating")

    with pytest.raises(ValueError):
        recall_at_k(rating_true, rating_pred, col_prediction="not_prediction")

    with pytest.raises(ValueError):
        ndcg_at_k(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        map_at_k(
            rating_pred,
            rating_pred,
            col_rating=DEFAULT_PREDICTION_COL,
            col_user="not_user",
        )
