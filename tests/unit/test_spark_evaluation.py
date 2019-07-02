# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import pytest
from reco_utils.evaluation.python_evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
)

try:
    from reco_utils.evaluation.spark_evaluation import (
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
