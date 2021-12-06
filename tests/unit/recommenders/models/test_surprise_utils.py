# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


try:
    import pandas as pd
    import pytest
    import surprise

    from recommenders.utils.constants import (
        DEFAULT_USER_COL,
        DEFAULT_ITEM_COL,
        DEFAULT_RATING_COL,
    )
    from recommenders.models.surprise.surprise_utils import (
        predict,
        compute_ranking_predictions,
    )
except:
    pass    # skip if experimental not installed

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


@pytest.mark.experimental
def test_predict(rating_true):
    svd = surprise.SVD()
    train_set = surprise.Dataset.load_from_df(
        rating_true, reader=surprise.Reader()
    ).build_full_trainset()
    svd.fit(train_set)

    preds = predict(svd, rating_true)
    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    user = rating_true.iloc[0]["userID"]
    item = rating_true.iloc[0]["itemID"]
    assert preds[(preds["userID"] == user) & (preds["itemID"] == item)][
        "prediction"
    ].values == pytest.approx(svd.predict(user, item).est, rel=TOL)

    preds = predict(
        svd,
        rating_true.rename(columns={"userID": "uid", "itemID": "iid"}),
        usercol="uid",
        itemcol="iid",
        predcol="pred",
    )
    assert set(preds.columns) == {"uid", "iid", "pred"}
    assert preds["uid"].dtypes == rating_true["userID"].dtypes
    assert preds["iid"].dtypes == rating_true["itemID"].dtypes
    user = rating_true.iloc[1]["userID"]
    item = rating_true.iloc[1]["itemID"]
    assert preds[(preds["uid"] == user) & (preds["iid"] == item)][
        "pred"
    ].values == pytest.approx(svd.predict(user, item).est, rel=TOL)


@pytest.mark.experimental
def test_recommend_k_items(rating_true):
    n_users = len(rating_true["userID"].unique())
    n_items = len(rating_true["itemID"].unique())
    svd = surprise.SVD()
    train_set = surprise.Dataset.load_from_df(
        rating_true, reader=surprise.Reader()
    ).build_full_trainset()
    svd.fit(train_set)

    preds = compute_ranking_predictions(svd, rating_true, remove_seen=True)
    assert set(preds.columns) == {"userID", "itemID", "prediction"}
    assert preds["userID"].dtypes == rating_true["userID"].dtypes
    assert preds["itemID"].dtypes == rating_true["itemID"].dtypes
    user = preds.iloc[0]["userID"]
    item = preds.iloc[0]["itemID"]
    assert preds[(preds["userID"] == user) & (preds["itemID"] == item)][
        "prediction"
    ].values == pytest.approx(svd.predict(user, item).est, rel=TOL)
    # Test default remove_seen=True
    assert pd.merge(rating_true, preds, on=["userID", "itemID"]).shape[0] == 0
    assert preds.shape[0] == (n_users * n_items - rating_true.shape[0])

    preds = compute_ranking_predictions(
        svd,
        rating_true.rename(columns={"userID": "uid", "itemID": "iid", "rating": "r"}),
        usercol="uid",
        itemcol="iid",
        predcol="pred",
        remove_seen=False,
    )
    assert set(preds.columns) == {"uid", "iid", "pred"}
    assert preds["uid"].dtypes == rating_true["userID"].dtypes
    assert preds["iid"].dtypes == rating_true["itemID"].dtypes
    user = preds.iloc[1]["uid"]
    item = preds.iloc[1]["iid"]
    assert preds[(preds["uid"] == user) & (preds["iid"] == item)][
        "pred"
    ].values == pytest.approx(svd.predict(user, item).est, rel=TOL)

    # Test remove_seen=False
    assert (
        pd.merge(
            rating_true, preds, left_on=["userID", "itemID"], right_on=["uid", "iid"]
        ).shape[0]
        == rating_true.shape[0]
    )
    assert preds.shape[0] == n_users * n_items
