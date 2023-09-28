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


TOL = 0.0001


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
        "map_at_k": pytest.approx(0.23613, TOL),
        "recall": pytest.approx(0.37777, TOL),
        "c_coverage": pytest.approx(0.8, TOL),
        "d_coverage": pytest.approx(1.9183, TOL),
        "item_novelty": pd.DataFrame(
            dict(ItemId=[1, 2, 3, 4, 5], item_novelty=[3.0, 3.0, 2.0, 1.41504, 3.0])
        ),
        "novelty": pytest.approx(2.83333, TOL),
        # diversity when using item co-occurrence count to calculate item similarity
        "diversity": pytest.approx(0.43096, TOL),
        "user_diversity": pd.DataFrame(
            dict(UserId=[1, 2, 3], user_diversity=[0.29289, 1.0, 0.0])
        ),
        # diversity values when using item features to calculate item similarity
        "diversity_item_feature_vector": pytest.approx(0.5000, TOL),
        "user_diversity_item_feature_vector": pd.DataFrame(
            dict(UserId=[1, 2, 3], user_diversity=[0.5000, 0.5000, 0.5000])
        ),
        "user_item_serendipity": pd.DataFrame(
            dict(
                UserId=[1, 1, 2, 2, 3, 3],
                ItemId=[3, 5, 2, 5, 1, 2],
                user_item_serendipity=[
                    0.72783,
                    0.0,
                    0.71132,
                    0.35777,
                    0.80755,
                    0.0,
                ],
            )
        ),
        "user_serendipity": pd.DataFrame(
            dict(UserId=[1, 2, 3], user_serendipity=[0.363915, 0.53455, 0.403775])
        ),
        "serendipity": pytest.approx(0.43408, TOL),
        # serendipity values when using item features to calculate item similarity
        "user_item_serendipity_item_feature_vector": pd.DataFrame(
            dict(
                UserId=[1, 1, 2, 2, 3, 3],
                ItemId=[3, 5, 2, 5, 1, 2],
                user_item_serendipity=[
                    0.5000,
                    0.0,
                    0.75,
                    0.5000,
                    0.6667,
                    0.0,
                ],
            )
        ),
        "user_serendipity_item_feature_vector": pd.DataFrame(
            dict(UserId=[1, 2, 3], user_serendipity=[0.2500, 0.625, 0.3333])
        ),
        "serendipity_item_feature_vector": pytest.approx(0.4028, TOL),
    }

# fmt: on
