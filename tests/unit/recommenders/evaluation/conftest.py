# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)


# fmt: off
@pytest.fixture
def rating_true():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
            DEFAULT_ITEM_COL: [3, 1, 4, 5, 6, 7, 2, 5, 6, 8, 9, 10, 11, 12, 13, 14, 1, 2],
            DEFAULT_RATING_COL: [3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1, 5, 4],
        }
    )


@pytest.fixture
def rating_pred():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
            DEFAULT_ITEM_COL: [12, 10, 3, 5, 11, 13, 4, 10, 7, 13, 1, 3, 5, 2, 11, 14, 3, 10],
            DEFAULT_PREDICTION_COL: [12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 14, 13],
            DEFAULT_RATING_COL: [3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3, 3, 3, 2, 1, 5, 4],
        }
    )


@pytest.fixture
def rating_nohit():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1],
            DEFAULT_ITEM_COL: [100] * 18,
            DEFAULT_PREDICTION_COL: [12, 14, 13, 12, 11, 10, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 14, 13],
        }
    )


@pytest.fixture
def diversity_data():
    train_df = pd.DataFrame(
        {"UserId": [1, 1, 1, 2, 2, 3, 3, 3], "ItemId": [1, 2, 4, 3, 4, 3, 4, 5]}
    )

    reco_df = pd.DataFrame(
        {
            "UserId": [1, 1, 2, 2, 3, 3],
            "ItemId": [3, 5, 2, 5, 1, 2],
            "Relevance": [1, 0, 1, 1, 1, 0],
        }
    )

    item_feature_df = pd.DataFrame(
        {
            "ItemId": [1, 2, 3, 4, 5],
            "features": [
                np.array([0.0, 1.0, 1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=float),
                np.array([0.0, 0.0, 0.0, 1.0, 1.0], dtype=float),
            ],
        }
    )

    return train_df, reco_df, item_feature_df
# fmt: on
