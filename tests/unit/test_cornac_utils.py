# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import pytest
import cornac

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)
from reco_utils.recommender.cornac.cornac_utils import (
    predict_rating,
    predict_ranking,
)
from reco_utils.evaluation.python_evaluation import (
    mae,
    rmse,
    ndcg_at_k,
    recall_at_k,
)

TOL = 0.001


@pytest.fixture
def rating_true():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            DEFAULT_ITEM_COL: [
                1,
                2,
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
            ],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1],
        }
    )


def test_predict_rating(rating_true):
    train_set = cornac.data.Dataset.from_uir(rating_true.itertuples(index=False), seed=42)
    mf = cornac.models.MF(k=100, max_iter=10000, seed=42).fit(train_set)

    preds = predict_rating(mf, rating_true)

    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    assert .02 > mae(rating_true, preds)  # ~0.018
    assert .03 > rmse(rating_true, preds)  # ~0.021


def test_predict_ranking(rating_true):
    train_set = cornac.data.Dataset.from_uir(rating_true.itertuples(index=False), seed=42)
    bpr = cornac.models.BPR(k=100, max_iter=10000, seed=42).fit(train_set)

    preds = predict_ranking(bpr, rating_true, remove_seen=False)

    n_users = len(rating_true["userID"].unique())
    n_items = len(rating_true["itemID"].unique())
    assert preds.shape[0] == n_users * n_items

    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    # perfect ranking achieved
    assert 1e-10 > 1 - ndcg_at_k(rating_true, preds)
    assert 1e-10 > 1 - recall_at_k(rating_true, preds)
