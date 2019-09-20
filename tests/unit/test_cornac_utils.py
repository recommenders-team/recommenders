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
    mf = cornac.models.MF()
    train_set = cornac.data.Dataset.from_uir(rating_true.itertuples(index=False), seed=42)
    mf.fit(train_set)

    preds = predict_rating(mf, rating_true)
    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    user = rating_true.iloc[0]["userID"]
    item = rating_true.iloc[0]["itemID"]
    assert preds[(preds["userID"] == user) & (preds["itemID"] == item)][
               "prediction"
           ].values == pytest.approx(mf.rate(train_set.uid_map[user],
                                             train_set.iid_map[item]).item(),
                                     rel=TOL)

    preds = predict_rating(
        mf,
        rating_true,
        usercol="userID",
        itemcol="itemID",
        predcol="prediction",
    )
    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    user = rating_true.iloc[1]["userID"]
    item = rating_true.iloc[1]["itemID"]
    assert preds[(preds["userID"] == user) & (preds["itemID"] == item)][
               "prediction"
           ].values == pytest.approx(mf.rate(train_set.uid_map[user],
                                             train_set.iid_map[item]).item(),
                                     rel=TOL)


def test_predict_ranking(rating_true):
    n_users = len(rating_true["userID"].unique())
    n_items = len(rating_true["itemID"].unique())
    mf = cornac.models.MF()
    train_set = cornac.data.Dataset.from_uir(rating_true.itertuples(index=False), seed=42)
    mf.fit(train_set)

    preds = predict_ranking(mf, rating_true, remove_seen=True)
    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    user = preds.iloc[0]["userID"]
    item = preds.iloc[0]["itemID"]
    assert preds[(preds["userID"] == user) & (preds["itemID"] == item)][
               "prediction"
           ].values == pytest.approx(mf.rate(train_set.uid_map[user],
                                             train_set.iid_map[item]).item(),
                                     rel=TOL)
    # Test default remove_seen=True
    assert pd.merge(rating_true, preds, on=["userID", "itemID"]).shape[0] == 0
    assert preds.shape[0] == (n_users * n_items - rating_true.shape[0])

    preds = predict_ranking(
        mf,
        rating_true,
        usercol="userID",
        itemcol="itemID",
        predcol="prediction",
        remove_seen=False,
    )
    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    user = preds.iloc[1]["userID"]
    item = preds.iloc[1]["itemID"]
    assert preds[(preds["userID"] == user) & (preds["itemID"] == item)][
               "prediction"
           ].values == pytest.approx(mf.rate(train_set.uid_map[user],
                                             train_set.iid_map[item]).item(),
                                     rel=TOL)

    # Test remove_seen=False
    assert (
            pd.merge(
                rating_true, preds, left_on=["userID", "itemID"], right_on=["userID", "itemID"]
            ).shape[0]
            == rating_true.shape[0]
    )
    assert preds.shape[0] == n_users * n_items
