# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal

from recommenders.evaluation.python_evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
)

try:
    from pyspark.sql import Row
    from recommenders.evaluation.spark_evaluation import (
        SparkDiversityEvaluation,
        SparkRankingEvaluation,
        SparkRatingEvaluation,
    )
except ImportError:
    pass  # skip this import if we are in pure python environment


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
    return rating_true, rating_pred


@pytest.fixture(scope="module")
def spark_data(python_data, spark):
    rating_true, rating_pred = python_data

    df_true = spark.createDataFrame(rating_true)
    df_pred = spark.createDataFrame(rating_pred)

    return df_true, df_pred


@pytest.fixture(scope="module")
def spark_diversity_data(spark):
    train_df = spark.createDataFrame(
        [
            Row(UserId=1, ItemId=1),
            Row(UserId=1, ItemId=2),
            Row(UserId=1, ItemId=4),
            Row(UserId=2, ItemId=3),
            Row(UserId=2, ItemId=4),
            Row(UserId=3, ItemId=3),
            Row(UserId=3, ItemId=4),
            Row(UserId=3, ItemId=5),
        ]
    )
    reco_df = spark.createDataFrame(
        [
            Row(UserId=1, ItemId=3, Relevance=1),
            Row(UserId=1, ItemId=5, Relevance=0),
            Row(UserId=2, ItemId=2, Relevance=1),
            Row(UserId=2, ItemId=5, Relevance=1),
            Row(UserId=3, ItemId=1, Relevance=1),
            Row(UserId=3, ItemId=2, Relevance=0),
        ]
    )
    return train_df, reco_df


@pytest.mark.spark
def test_init_spark(spark):
    assert spark is not None


@pytest.mark.spark
def test_init_spark_rating_eval(spark_data):
    df_true, df_pred = spark_data
    evaluator = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator is not None


@pytest.mark.spark
def test_spark_rmse(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator1 = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator1.rmse() == 0

    evaluator2 = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator2.rmse() == target_metrics["rmse"]


@pytest.mark.spark
def test_spark_mae(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator1 = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator1.mae() == 0

    evaluator2 = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator2.mae() == target_metrics["mae"]


@pytest.mark.spark
def test_spark_rsquared(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator1 = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator1.rsquared() == pytest.approx(1.0, TOL)

    evaluator2 = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator2.rsquared() == target_metrics["rsquared"]


@pytest.mark.spark
def test_spark_exp_var(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator1 = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator1.exp_var() == pytest.approx(1.0, TOL)

    evaluator2 = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator2.exp_var() == target_metrics["exp_var"]


@pytest.mark.spark
def test_spark_recall(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_pred)
    assert evaluator.recall_at_k() == target_metrics["recall"]

    evaluator1 = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator1.recall_at_k() == target_metrics["recall"]


@pytest.mark.spark
def test_spark_precision(spark_data, target_metrics, spark):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.precision_at_k() == target_metrics["precision"]

    evaluator1 = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator1.precision_at_k() == target_metrics["precision"]

    # Check normalization
    single_user = pd.DataFrame(
        {"userID": [1, 1, 1], "itemID": [1, 2, 3], "rating": [5, 4, 3]}
    )
    df_single = spark.createDataFrame(single_user)
    evaluator2 = SparkRankingEvaluation(
        df_single, df_single, k=3, col_prediction="rating"
    )
    assert evaluator2.precision_at_k() == 1

    same_items = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 2],
            "itemID": [1, 2, 3, 1, 2, 3],
            "rating": [5, 4, 3, 5, 5, 3],
        }
    )
    df_same = spark.createDataFrame(same_items)
    evaluator3 = SparkRankingEvaluation(df_same, df_same, k=3, col_prediction="rating")
    assert evaluator3.precision_at_k() == 1

    # Check that if the sample size is smaller than k, the maximum precision can not be 1
    # if we do precision@5 when there is only 3 items, we can get a maximum of 3/5.
    evaluator4 = SparkRankingEvaluation(df_same, df_same, k=5, col_prediction="rating")
    assert evaluator4.precision_at_k() == 0.6


@pytest.mark.spark
def test_spark_ndcg(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator0 = SparkRankingEvaluation(df_true, df_true, k=10, col_prediction="rating")
    assert evaluator0.ndcg_at_k() == 1.0

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.ndcg_at_k() == target_metrics["ndcg"]

    evaluator1 = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator1.ndcg_at_k() == target_metrics["ndcg"]


@pytest.mark.spark
def test_spark_map(spark_data, target_metrics):
    df_true, df_pred = spark_data

    evaluator1 = SparkRankingEvaluation(
        k=10, rating_true=df_true, rating_pred=df_true, col_prediction="rating"
    )
    assert evaluator1.map_at_k() == 1.0

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.map_at_k() == target_metrics["map"]

    evaluator1 = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator1.map_at_k() == target_metrics["map"]


@pytest.mark.spark
def test_spark_python_match(python_data, spark):
    # Test on the original data with k = 10.
    df_true, df_pred = python_data

    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark1 = SparkRankingEvaluation(dfs_true, dfs_pred, k=10)

    assert recall_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark1.recall_at_k(), TOL
    )
    assert precision_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark1.precision_at_k(), TOL
    )
    assert ndcg_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark1.ndcg_at_k(), TOL
    )
    assert map_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark1.map_at_k(), TOL
    )

    # Test on the original data with k = 3.
    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark2 = SparkRankingEvaluation(dfs_true, dfs_pred, k=3)

    assert recall_at_k(df_true, df_pred, k=3) == pytest.approx(
        eval_spark2.recall_at_k(), TOL
    )
    assert precision_at_k(df_true, df_pred, k=3) == pytest.approx(
        eval_spark2.precision_at_k(), TOL
    )
    assert ndcg_at_k(df_true, df_pred, k=3) == pytest.approx(
        eval_spark2.ndcg_at_k(), TOL
    )
    assert map_at_k(df_true, df_pred, k=3) == pytest.approx(eval_spark2.map_at_k(), TOL)

    # Remove the first row from the original data.
    df_pred = df_pred[1:-1]

    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark3 = SparkRankingEvaluation(dfs_true, dfs_pred, k=10)

    assert recall_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark3.recall_at_k(), TOL
    )
    assert precision_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark3.precision_at_k(), TOL
    )
    assert ndcg_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark3.ndcg_at_k(), TOL
    )
    assert map_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark3.map_at_k(), TOL
    )

    # Test with one user
    df_pred = df_pred.loc[df_pred["userID"] == 3]
    df_true = df_true.loc[df_true["userID"] == 3]

    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark4 = SparkRankingEvaluation(dfs_true, dfs_pred, k=10)

    assert recall_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark4.recall_at_k(), TOL
    )
    assert precision_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark4.precision_at_k(), TOL
    )
    assert ndcg_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark4.ndcg_at_k(), TOL
    )
    assert map_at_k(df_true, df_pred, k=10) == pytest.approx(
        eval_spark4.map_at_k(), TOL
    )


@pytest.mark.spark
def test_catalog_coverage(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    c_coverage = evaluator.catalog_coverage()
    assert c_coverage == target_metrics["c_coverage"]


@pytest.mark.spark
def test_distributional_coverage(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    d_coverage = evaluator.distributional_coverage()
    assert d_coverage == target_metrics["d_coverage"]


@pytest.mark.spark
def test_item_novelty(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    actual = evaluator.historical_item_novelty().toPandas()
    assert_frame_equal(
        target_metrics["item_novelty"], actual, check_exact=False, check_less_precise=4
    )
    assert np.all(actual["item_novelty"].values >= 0)
    # Test that novelty is zero when data includes only one item
    train_df_new = train_df.filter("ItemId == 3")
    evaluator = SparkDiversityEvaluation(
        train_df=train_df_new, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    actual = evaluator.historical_item_novelty().toPandas()
    assert actual["item_novelty"].values[0] == 0


@pytest.mark.spark
def test_novelty(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    novelty = evaluator.novelty()
    assert target_metrics["novelty"] == novelty
    assert novelty >= 0
    # Test that novelty is zero when data includes only one item
    train_df_new = train_df.filter("ItemId == 3")
    reco_df_new = reco_df.filter("ItemId == 3")
    evaluator = SparkDiversityEvaluation(
        train_df=train_df_new, reco_df=reco_df_new, col_user="UserId", col_item="ItemId"
    )
    assert evaluator.novelty() == 0


@pytest.mark.spark
def test_user_diversity(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    actual = evaluator.user_diversity().toPandas()
    assert_frame_equal(
        target_metrics["user_diversity"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


@pytest.mark.spark
def test_diversity(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert target_metrics["diversity"] == evaluator.diversity()


@pytest.mark.spark
def test_user_item_serendipity(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    actual = evaluator.user_item_serendipity().toPandas()
    assert_frame_equal(
        target_metrics["user_item_serendipity"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


@pytest.mark.spark
def test_user_serendipity(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    actual = evaluator.user_serendipity().toPandas()
    assert_frame_equal(
        target_metrics["user_serendipity"],
        actual,
        check_exact=False,
        check_less_precise=4,
    )


@pytest.mark.spark
def test_serendipity(spark_diversity_data, target_metrics):
    train_df, reco_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    assert target_metrics["serendipity"] == evaluator.serendipity()
