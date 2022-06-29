# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock
from sklearn.preprocessing import minmax_scale
from pandas.util.testing import assert_frame_equal

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)
from recommenders.evaluation.python_evaluation import (
    _check_column_dtypes,
    merge_rating_true_pred,
    merge_ranking_true_pred,
    rmse,
    mae,
    rsquared,
    exp_var,
    get_top_k_items,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    auc,
    logloss,
    user_diversity,
    diversity,
    historical_item_novelty,
    novelty,
    user_item_serendipity,
    user_serendipity,
    serendipity,
    catalog_coverage,
    distributional_coverage,
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
# fmt: on


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
        _check_column_dtypes(Mock())(
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


def test_get_top_k_items(rating_true):
    top_3_items_df = get_top_k_items(
        dataframe=rating_true,
        col_user=DEFAULT_USER_COL,
        col_rating=DEFAULT_RATING_COL,
        k=3,
    )
    top_3_user_true = pd.Series([1, 1, 1, 2, 2, 2, 3, 3, 3])
    top_3_rating_true = pd.Series([5, 4, 3, 5, 5, 3, 5, 5, 5])
    top_3_rank_true = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3])
    assert top_3_items_df[DEFAULT_USER_COL].equals(top_3_user_true)
    assert top_3_items_df[DEFAULT_RATING_COL].equals(top_3_rating_true)
    assert top_3_items_df["rank"].equals(top_3_rank_true)
    assert top_3_items_df[DEFAULT_ITEM_COL][:3].equals(pd.Series([1, 2, 3]))
    # First two itemIDs of user 2. The scores are both 5, so any order is OK.
    assert set(top_3_items_df[DEFAULT_ITEM_COL][3:5]) == set([1, 4])
    # Third itemID of user 2. Both item 5 and 6 have a score of 3, so either one is OK.
    assert top_3_items_df[DEFAULT_ITEM_COL][5] in [5, 6]
    # All itemIDs of user 3. All three items have a score of 5, so any order is OK.
    assert set(top_3_items_df[DEFAULT_ITEM_COL][6:]) == set([2, 5, 6])


# Test get_top_k_items() when k is larger than the number of available items
def test_get_top_k_items_largek(rating_true):
    top_6_items_df = get_top_k_items(
        dataframe=rating_true,
        col_user=DEFAULT_USER_COL,
        col_rating=DEFAULT_RATING_COL,
        k=6,
    )
    top_6_user_true = pd.Series([1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])
    top_6_rating_true = pd.Series([5, 4, 3, 5, 5, 3, 3, 1, 5, 5, 5, 4, 4, 3])
    top_6_rank_true = pd.Series([1, 2, 3, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6])
    assert top_6_items_df[DEFAULT_USER_COL].equals(top_6_user_true)
    assert top_6_items_df[DEFAULT_RATING_COL].equals(top_6_rating_true)
    assert top_6_items_df["rank"].equals(top_6_rank_true)
    assert top_6_items_df[DEFAULT_ITEM_COL][:3].equals(pd.Series([1, 2, 3]))
    # First two itemIDs of user 2. The scores are both 5, so any order is OK.
    assert set(top_6_items_df[DEFAULT_ITEM_COL][3:5]) == set([1, 4])
    # Third and fourth itemID of user 2. The scores are both 3, so any order is OK.
    assert set(top_6_items_df[DEFAULT_ITEM_COL][5:7]) == set([5, 6])
    assert top_6_items_df[DEFAULT_ITEM_COL][7] == 7
    # First three itemIDs of user 3. The scores are both 5, so any order is OK.
    assert set(top_6_items_df[DEFAULT_ITEM_COL][8:11]) == set([2, 5, 6])
    # Fourth and fifth itemID of user 3. The scores are both 4, so any order is OK.
    assert set(top_6_items_df[DEFAULT_ITEM_COL][11:13]) == set([8, 9])
    # Sixth itemID of user 3. Item 10,11,12 have a score of 3, so either one is OK.
    assert top_6_items_df[DEFAULT_ITEM_COL][13] in [10, 11, 12]


def test_python_ndcg_at_k(rating_true, rating_pred, rating_nohit):
    assert ndcg_at_k(
        rating_true=rating_true,
        rating_pred=rating_true,
        col_prediction=DEFAULT_RATING_COL,
        k=10,
    ) == pytest.approx(1.0, TOL)
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


# test diversity metrics
@pytest.fixture(scope="module")
def target_metrics():
    return {
        "c_coverage": pytest.approx(0.8, TOL),
        "d_coverage": pytest.approx(1.9183, TOL),
        "item_novelty": pd.DataFrame(
            dict(ItemId=[1, 2, 3, 4, 5], item_novelty=[3.0, 3.0, 2.0, 1.41504, 3.0])
        ),
        "novelty": pytest.approx(2.83333, TOL),
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


@pytest.fixture(scope="module")
def python_diversity_data():
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


def test_catalog_coverage(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    c_coverage = catalog_coverage(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert c_coverage == target_metrics["c_coverage"]


def test_distributional_coverage(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    d_coverage = distributional_coverage(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert d_coverage == target_metrics["d_coverage"]


def test_item_novelty(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    actual = historical_item_novelty(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert_frame_equal(
        target_metrics["item_novelty"], actual, check_exact=False, check_less_precise=4
    )
    assert np.all(actual["item_novelty"].values >= 0)
    # Test that novelty is zero when data includes only one item
    train_df_new = train_df.loc[train_df["ItemId"] == 3]
    actual = historical_item_novelty(
        train_df=train_df_new, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert actual["item_novelty"].values[0] == 0


def test_novelty(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    actual = novelty(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert target_metrics["novelty"] == actual
    assert actual >= 0
    # Test that novelty is zero when data includes only one item
    train_df_new = train_df.loc[train_df["ItemId"] == 3]
    reco_df_new = reco_df.loc[reco_df["ItemId"] == 3]
    assert (
        novelty(
            train_df=train_df_new,
            reco_df=reco_df_new,
            col_user="UserId",
            col_item="ItemId",
        )
        == 0
    )


def test_user_diversity(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    actual = user_diversity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=None,
        item_sim_measure="item_cooccurrence_count",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance=None,
    )
    assert_frame_equal(
        target_metrics["user_diversity"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


def test_diversity(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    assert target_metrics["diversity"] == diversity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=None,
        item_sim_measure="item_cooccurrence_count",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance=None,
    )


def test_user_item_serendipity(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    actual = user_item_serendipity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=None,
        item_sim_measure="item_cooccurrence_count",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance="Relevance",
    )
    assert_frame_equal(
        target_metrics["user_item_serendipity"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


def test_user_serendipity(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    actual = user_serendipity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=None,
        item_sim_measure="item_cooccurrence_count",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance="Relevance",
    )
    assert_frame_equal(
        target_metrics["user_serendipity"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


def test_serendipity(python_diversity_data, target_metrics):
    train_df, reco_df, _ = python_diversity_data
    assert target_metrics["serendipity"] == serendipity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=None,
        item_sim_measure="item_cooccurrence_count",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance="Relevance",
    )


def test_user_diversity_item_feature_vector(python_diversity_data, target_metrics):
    train_df, reco_df, item_feature_df = python_diversity_data
    actual = user_diversity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance=None,
    )
    assert_frame_equal(
        target_metrics["user_diversity_item_feature_vector"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


def test_diversity_item_feature_vector(python_diversity_data, target_metrics):
    train_df, reco_df, item_feature_df = python_diversity_data
    assert target_metrics["diversity_item_feature_vector"] == diversity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance=None,
    )


def test_user_item_serendipity_item_feature_vector(
    python_diversity_data, target_metrics
):
    train_df, reco_df, item_feature_df = python_diversity_data
    actual = user_item_serendipity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance="Relevance",
    )
    assert_frame_equal(
        target_metrics["user_item_serendipity_item_feature_vector"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


def test_user_serendipity_item_feature_vector(python_diversity_data, target_metrics):
    train_df, reco_df, item_feature_df = python_diversity_data
    actual = user_serendipity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance="Relevance",
    )
    assert_frame_equal(
        target_metrics["user_serendipity_item_feature_vector"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


def test_serendipity_item_feature_vector(python_diversity_data, target_metrics):
    train_df, reco_df, item_feature_df = python_diversity_data
    assert target_metrics["serendipity_item_feature_vector"] == serendipity(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_sim="sim",
        col_relevance="Relevance",
    )
