# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from recommenders.evaluation.python_evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    map,
)

try:
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql.types import ArrayType, FloatType, IntegerType, StructField, StructType
    from recommenders.evaluation.spark_evaluation import (
        SparkDiversityEvaluation,
        SparkRankingEvaluation,
        SparkRatingEvaluation,
    )
except ImportError:
    pass  # skip this import if we are in pure python environment


TOL = 0.0001


@pytest.fixture
def spark_data(rating_true, rating_pred, spark):
    df_true = spark.createDataFrame(rating_true)
    df_pred = spark.createDataFrame(rating_pred)

    return df_true, df_pred


@pytest.fixture
def spark_diversity_data(diversity_data, spark):
    train_df, reco_df, item_feature_df = diversity_data
    
    train_df = spark.createDataFrame(train_df)
    reco_df = spark.createDataFrame(reco_df)
    item_feature_df["features"] = item_feature_df["features"].apply(lambda x: x.tolist())
    field = [
        StructField("ItemId", IntegerType(), True),
        StructField("features", ArrayType(FloatType()), True),
    ]
    item_feature_df = spark.createDataFrame(item_feature_df, schema=StructType(field))
    # Array[Float] to VectorUDT
    item_feature_df = item_feature_df.withColumn("features", array_to_vector(item_feature_df["features"]))

    return train_df, reco_df, item_feature_df


@pytest.mark.spark
def test_init_spark(spark):
    assert spark is not None


@pytest.mark.spark
def test_init_spark_rating_eval(spark_data):
    df_true, df_pred = spark_data
    evaluator = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator is not None


@pytest.mark.spark
def test_spark_rmse(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator.rmse() == 0

    evaluator = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator.rmse() == pytest.approx(7.254309, TOL)


@pytest.mark.spark
def test_spark_mae(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator.mae() == 0

    evaluator = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator.mae() == pytest.approx(6.375, TOL)


@pytest.mark.spark
def test_spark_rsquared(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator.rsquared() == pytest.approx(1.0, TOL)

    evaluator = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator.rsquared() == pytest.approx(-31.699029, TOL)


@pytest.mark.spark
def test_spark_exp_var(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
    assert evaluator.exp_var() == pytest.approx(1.0, TOL)

    evaluator = SparkRatingEvaluation(df_true, df_pred)
    assert evaluator.exp_var() == pytest.approx(-6.4466, TOL)


@pytest.mark.spark
def test_spark_recall_at_k(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_pred)
    assert evaluator.recall_at_k() == pytest.approx(0.37777, TOL)

    evaluator = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator.recall_at_k() == pytest.approx(0.37777, TOL)


@pytest.mark.spark
def test_spark_precision_at_k(spark_data, spark):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.precision_at_k() == pytest.approx(0.26666, TOL)

    evaluator = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator.precision_at_k() == pytest.approx(0.26666, TOL)

    # Check normalization
    single_user = pd.DataFrame(
        {"userID": [1, 1, 1], "itemID": [1, 2, 3], "rating": [5, 4, 3]}
    )
    df_single = spark.createDataFrame(single_user)
    evaluator = SparkRankingEvaluation(
        df_single, df_single, k=3, col_prediction="rating"
    )
    assert evaluator.precision_at_k() == 1

    same_items = pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 2],
            "itemID": [1, 2, 3, 1, 2, 3],
            "rating": [5, 4, 3, 5, 5, 3],
        }
    )
    df_same = spark.createDataFrame(same_items)
    evaluator = SparkRankingEvaluation(df_same, df_same, k=3, col_prediction="rating")
    assert evaluator.precision_at_k() == 1

    # Check that if the sample size is smaller than k, the maximum precision can not be 1
    # if we do precision@5 when there is only 3 items, we can get a maximum of 3/5.
    evaluator = SparkRankingEvaluation(df_same, df_same, k=5, col_prediction="rating")
    assert evaluator.precision_at_k() == 0.6


@pytest.mark.spark
def test_spark_ndcg_at_k(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_true, k=10, col_prediction="rating")
    assert evaluator.ndcg_at_k() == 1.0

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.ndcg_at_k() == pytest.approx(0.38172, TOL)

    evaluator = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator.ndcg_at_k() == pytest.approx(0.38172, TOL)


@pytest.mark.spark
def test_spark_map(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_true, k=10, col_prediction="rating")
    assert evaluator.map() == 1.0

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.map() == pytest.approx(0.23613, TOL)

    evaluator = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator.map() == pytest.approx(0.23613, TOL)


@pytest.mark.spark
def test_spark_map_at_k(spark_data):
    df_true, df_pred = spark_data

    evaluator = SparkRankingEvaluation(df_true, df_true, k=10, col_prediction="rating")
    assert evaluator.map_at_k() == 1.0

    evaluator = SparkRankingEvaluation(df_true, df_pred, k=10)
    assert evaluator.map_at_k() == pytest.approx(0.23613, TOL)

    evaluator = SparkRankingEvaluation(
        df_true, df_pred, relevancy_method="by_threshold", threshold=3.5
    )
    assert evaluator.map_at_k() == pytest.approx(0.23613, TOL)


@pytest.mark.spark
@pytest.mark.parametrize(
    "k,pred_start_row_i,user_id",
    [
        (10, 0, None),
        (3, 0, None),   # Different k
        (10, 0, None),  # Different pred
        (10, 0, 3),     # Test with one user (userID == 3)
    ]
)
def test_spark_python_match(rating_true, rating_pred, spark, k, pred_start_row_i, user_id):
    df_true, df_pred = rating_true, rating_pred
    df_pred = df_pred[pred_start_row_i:]
    if user_id is not None:
        df_pred = df_pred.loc[df_pred["userID"] == 3]
        df_true = df_true.loc[df_true["userID"] == 3]

    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    # Test on the original data with k = 10.
    evaluator = SparkRankingEvaluation(dfs_true, dfs_pred, k=k)

    assert recall_at_k(df_true, df_pred, k=k) == pytest.approx(
        evaluator.recall_at_k(), TOL
    )
    assert precision_at_k(df_true, df_pred, k=k) == pytest.approx(
        evaluator.precision_at_k(), TOL
    )
    assert ndcg_at_k(df_true, df_pred, k=k) == pytest.approx(
        evaluator.ndcg_at_k(), TOL
    )
    assert map_at_k(df_true, df_pred, k=k) == pytest.approx(
        evaluator.map_at_k(), TOL
    )
    assert map(df_true, df_pred, k=k) == pytest.approx(
        evaluator.map(), TOL
    )


@pytest.mark.spark
def test_catalog_coverage(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    c_coverage = evaluator.catalog_coverage()
    assert c_coverage == pytest.approx(0.8, TOL)


@pytest.mark.spark
def test_distributional_coverage(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    d_coverage = evaluator.distributional_coverage()
    assert d_coverage == pytest.approx(1.9183, TOL)


@pytest.mark.spark
def test_item_novelty(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data

    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    actual = evaluator.historical_item_novelty().toPandas()
    assert_frame_equal(
        pd.DataFrame(
            dict(ItemId=[1, 2, 3, 4, 5], item_novelty=[3.0, 3.0, 2.0, 1.41504, 3.0])
        ),
        actual,
        check_exact=False,
        atol=TOL,
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
def test_novelty(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert evaluator.novelty() == pytest.approx(2.83333, TOL)

    # Test that novelty is zero when data includes only one item
    train_df_new = train_df.filter("ItemId == 3")
    reco_df_new = reco_df.filter("ItemId == 3")
    evaluator = SparkDiversityEvaluation(
        train_df=train_df_new, reco_df=reco_df_new, col_user="UserId", col_item="ItemId"
    )
    assert evaluator.novelty() == 0


@pytest.mark.spark
def test_user_diversity(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    actual = evaluator.user_diversity().toPandas()
    assert_frame_equal(
        pd.DataFrame(
            dict(UserId=[1, 2, 3], user_diversity=[0.29289, 1.0, 0.0])
        ),
        actual,
        check_exact=False,
        atol=TOL,
    )


@pytest.mark.spark
def test_diversity(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df, reco_df=reco_df, col_user="UserId", col_item="ItemId"
    )
    assert evaluator.diversity() == pytest.approx(0.43096, TOL)


@pytest.mark.spark
def test_user_item_serendipity(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    actual = evaluator.user_item_serendipity().toPandas()
    assert_frame_equal(
        pd.DataFrame(
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
        actual,
        check_exact=False,
        atol=TOL,
    )


@pytest.mark.spark
def test_user_serendipity(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    actual = evaluator.user_serendipity().toPandas()
    assert_frame_equal(
        pd.DataFrame(
            dict(UserId=[1, 2, 3], user_serendipity=[0.363915, 0.53455, 0.403775])
        ),
        actual,
        check_exact=False,
        atol=TOL,
    )


@pytest.mark.spark
def test_serendipity(spark_diversity_data):
    train_df, reco_df, _ = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    assert evaluator.serendipity() == pytest.approx(0.43408, TOL)


@pytest.mark.spark
def test_user_diversity_item_feature_vector(spark_diversity_data):
    train_df, reco_df, item_feature_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
    )
    actual = evaluator.user_diversity().toPandas()
    assert_frame_equal(
        pd.DataFrame(
            dict(UserId=[1, 2, 3], user_diversity=[0.5000, 0.5000, 0.5000])
        ),
        actual,
        check_exact=False,
        atol=TOL,
    )


@pytest.mark.spark
def test_diversity_item_feature_vector(spark_diversity_data):
    train_df, reco_df, item_feature_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
    )
    assert evaluator.diversity() == pytest.approx(0.5000, TOL)


@pytest.mark.spark
def test_user_item_serendipity_item_feature_vector(
    spark_diversity_data,
):
    train_df, reco_df, item_feature_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    actual = evaluator.user_item_serendipity().toPandas()
    assert_frame_equal(
        pd.DataFrame(
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
        actual,
        check_exact=False,
        atol=TOL,
    )


@pytest.mark.spark
def test_user_serendipity_item_feature_vector(spark_diversity_data):
    train_df, reco_df, item_feature_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    actual = evaluator.user_serendipity().toPandas()
    assert_frame_equal(
        pd.DataFrame(
            dict(UserId=[1, 2, 3], user_serendipity=[0.2500, 0.625, 0.3333])
        ),
        actual,
        check_exact=False,
        atol=TOL,
    )


@pytest.mark.spark
def test_serendipity_item_feature_vector(spark_diversity_data):
    train_df, reco_df, item_feature_df = spark_diversity_data
    evaluator = SparkDiversityEvaluation(
        train_df=train_df,
        reco_df=reco_df,
        item_feature_df=item_feature_df,
        item_sim_measure="item_feature_vector",
        col_user="UserId",
        col_item="ItemId",
        col_relevance="Relevance",
    )
    assert evaluator.serendipity() == pytest.approx(0.4028, TOL)
