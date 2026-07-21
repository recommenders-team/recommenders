# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
import cornac

from recommenders.models.cornac.bpr import BPR
from recommenders.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
    SEED,
)


@pytest.fixture
def train_data():
    # 3 users, 7 items; each user has seen 3 items, leaving at least 3 unseen
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 1, 1, 2, 2, 2, 3, 3, 3],
            DEFAULT_ITEM_COL: [1, 2, 3, 2, 3, 4, 4, 5, 6],
            DEFAULT_RATING_COL: [1.0] * 9,
        }
    )


@pytest.fixture
def trained_bpr(train_data):
    dataset = cornac.data.Dataset.from_uir(
        train_data.itertuples(index=False),
        seed=SEED,
    )
    model = BPR(k=10, max_iter=50, seed=SEED)
    model.fit(dataset)
    return model


def test_recommend_k_items_top_k_enforced(trained_bpr, train_data):
    top_k = 3
    preds = trained_bpr.recommend_k_items(train_data, top_k=top_k)
    assert set(preds.columns) == {DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL}
    for _, group in preds.groupby(DEFAULT_USER_COL):
        assert len(group) <= top_k


def test_recommend_k_items_remove_seen(trained_bpr, train_data):
    preds = trained_bpr.recommend_k_items(train_data, top_k=3, remove_seen=True)
    seen = set(zip(train_data[DEFAULT_USER_COL], train_data[DEFAULT_ITEM_COL]))
    pred_pairs = set(zip(preds[DEFAULT_USER_COL], preds[DEFAULT_ITEM_COL]))
    assert len(seen & pred_pairs) == 0


def test_recommend_k_items_no_inf_scores(trained_bpr, train_data):
    preds = trained_bpr.recommend_k_items(train_data, top_k=3, remove_seen=True)
    assert not np.isinf(preds[DEFAULT_PREDICTION_COL].values).any()
