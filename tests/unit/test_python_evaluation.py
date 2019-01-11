# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Test evaluation
"""
import pandas as pd
import pytest

from reco_utils.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
)

TOL = 0.0001


@pytest.fixture
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
    }


@pytest.fixture(scope="module")
def python_data():
    rating_true = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "itemID": [1, 2, 3, 1, 4, 5, 6, 7, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
            "rating": [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1],
        }
    )
    rating_pred = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "itemID": [3, 10, 12, 10, 3, 5, 11, 13, 4, 10, 7, 13, 1, 3, 5, 2, 11, 14],
            "prediction": [
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
            "userID": [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            "itemID": [100] * rating_pred.shape[0],
            "prediction": [
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

    return rating_true, rating_pred, rating_nohit


def test_python_rmse(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data
    assert (
        rmse(rating_true=rating_true, rating_pred=rating_true, col_prediction="rating")
        == 0
    )
    assert rmse(rating_true, rating_pred) == target_metrics["rmse"]


def test_python_mae(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data
    assert (
        mae(rating_true=rating_true, rating_pred=rating_true, col_prediction="rating")
        == 0
    )
    assert mae(rating_true, rating_pred) == target_metrics["mae"]


def test_python_rsquared(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data

    assert rsquared(
        rating_true=rating_true, rating_pred=rating_true, col_prediction="rating"
    ) == pytest.approx(1.0, TOL)

    assert rsquared(rating_true, rating_pred) == target_metrics["rsquared"]


def test_python_exp_var(python_data, target_metrics):
    rating_true, rating_pred, _ = python_data

    assert exp_var(
        rating_true=rating_true, rating_pred=rating_true, col_prediction="rating"
    ) == pytest.approx(1.0, TOL)

    assert exp_var(rating_true, rating_pred) == target_metrics["exp_var"]


def test_python_ndcg_at_k(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data

    assert (
        ndcg_at_k(
            k=10,
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction="rating",
        )
        == 1
    )
    assert ndcg_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert ndcg_at_k(rating_true, rating_pred, k=10) == target_metrics["ndcg"]


def test_python_map_at_k(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data

    assert (
        map_at_k(
            k=10,
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction="rating",
        )
        == 1
    )
    assert map_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert map_at_k(rating_true, rating_pred, k=10) == target_metrics["map"]


def test_python_precision(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data

    assert (
        precision_at_k(
            k=10,
            rating_true=rating_true,
            rating_pred=rating_true,
            col_prediction="rating",
        )
        == 0.6
    )
    assert precision_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert precision_at_k(rating_true, rating_pred, k=10) == target_metrics["precision"]


def test_python_recall(python_data, target_metrics):
    rating_true, rating_pred, rating_nohit = python_data

    assert recall_at_k(
        k=10, rating_true=rating_true, rating_pred=rating_true, col_prediction="rating"
    ) == pytest.approx(1, 0.1)
    assert recall_at_k(rating_true, rating_nohit, k=10) == 0.0
    assert recall_at_k(rating_true, rating_pred, k=10) == target_metrics["recall"]


def test_python_errors(python_data):
    rating_true, rating_pred, _ = python_data

    with pytest.raises(ValueError):
        rmse(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        mae(rating_pred, rating_pred, col_rating="prediction", col_user="not_user")

    with pytest.raises(ValueError):
        rsquared(rating_true, rating_pred, col_item="not_item")

    with pytest.raises(ValueError):
        exp_var(rating_pred, rating_pred, col_rating="prediction", col_item="not_item")

    with pytest.raises(ValueError):
        precision_at_k(rating_true, rating_pred, col_rating="not_rating")

    with pytest.raises(ValueError):
        recall_at_k(rating_true, rating_pred, col_prediction="not_prediction")

    with pytest.raises(ValueError):
        ndcg_at_k(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        map_at_k(rating_pred, rating_pred, col_rating="prediction", col_user="not_user")
