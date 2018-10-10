"""
Test evaluation
"""
import pandas as pd
import pytest

from utilities.evaluation.spark_evaluation import (
    SparkRankingEvaluation,
    SparkRatingEvaluation,
)
from utilities.common.spark_utils import start_or_get_spark


@pytest.fixture(scope="module")
def target_metrics():
    return {
        "rmse": pytest.approx(7.254309, 0.0001),
        "mae": pytest.approx(6.375, 0.0001),
        "rsquared": pytest.approx(-31.699029, 0.0001),
        "exp_var": pytest.approx(-6.4466, 0.01),
        "ndcg": pytest.approx(0.38172, 0.0001),
        "precision": pytest.approx(0.26666, 0.0001),
        "map": pytest.approx(0.23613, 0.0001),
        "recall": pytest.approx(0.37777, 0.0001),
    }


@pytest.fixture(scope="module")
def spark_data():
    """Get Python labels"""
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
    spark = start_or_get_spark("EvaluationTesting", "local")
    df_true = spark.createDataFrame(rating_true)
    df_pred = spark.createDataFrame(rating_pred)
    return df_true, df_pred


@pytest.mark.spark
def test_init_spark_rating_eval(spark_data):
    """Test initializer spark"""
    spark = start_or_get_spark()
    assert spark is not None
    # evaluator = SparkRatingEvaluation(df_true, df_pred)

    # assert evaluator is not None
    # assert evaluator.rating_true == df_true
    # assert evaluator.rating_pred == df_pred
    # assert evaluator.get_available_metrics() is not None


# @pytest.mark.spark
# def test_spark_rmse(spark_data, target_metrics):
#     """Test Spark evaluator RMSE"""
#     df_true, df_pred = spark_data
#     evaluator1 = SparkRatingEvaluation(df_true, df_true, col_prediction="rating")
#     assert evaluator1.rmse() == 0
#     evaluator2 = SparkRatingEvaluation(df_true, df_pred)
#     assert evaluator2.rmse() == target_metrics["rmse"]


#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_mae(spark_data, target_metrics):
#     """Test Spark evaluator MAE"""
#     df_true, df_pred = spark_data
#     evaluator1 = Evaluator(df_true, df_true, col_prediction="rating")
#     assert evaluator1.mae() == 0
#     evaluator2 = Evaluator(df_true, df_pred)
#     assert evaluator2.mae() == target_metrics['mae']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_rsquared(spark_data, target_metrics):
#     """Test Spark evaluator rsquared"""
#     df_true, df_pred = spark_data
#
#     evaluator1 = SparkRatingEvaluation(
#         df_true, df_true, col_prediction="rating")
#     assert evaluator1.rsquared() == pytest.approx(1.0, 0.0001)
#
#     evaluator2 = SparkRatingEvaluation(df_true, df_pred)
#     assert evaluator2.rsquared() == target_metrics['rsquared']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_exp_var(spark_data, target_metrics):
#     """Test Spark evaluator exp_var"""
#     df_true, df_pred = spark_data
#
#     evaluator1 = SparkRatingEvaluation(
#         df_true, df_true, col_prediction="rating")
#     assert evaluator1.exp_var() == pytest.approx(1.0, 0.0001)
#
#     evaluator2 = SparkRatingEvaluation(df_true, df_pred)
#     assert evaluator2.exp_var() == target_metrics['exp_var']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_recall(spark_data, target_metrics):
#     """Test Spark ranking evaluator recall."""
#     df_true, df_pred = spark_data
#
#     evaluator = Evaluator(df_true, df_pred)
#     assert evaluator.recall_at_k() == target_metrics['recall']
#
#     evaluator1 = Evaluator(df_true, df_pred,
#                            relevancy_method="by_threshold")
#     assert evaluator1.recall_at_k() == target_metrics['recall']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_precision(spark_data, target_metrics):
#     """Test Spark ranking evaluator precision."""
#     df_true, df_pred = spark_data
#
#     evaluator = Evaluator(df_true, df_pred, top_k=10)
#     assert evaluator.precision_at_k() == target_metrics['precision']
#
#     evaluator1 = Evaluator(df_true, df_pred,
#                            relevancy_method="by_threshold")
#     assert evaluator1.precision_at_k() == target_metrics['precision']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_ndcg(spark_data, target_metrics):
#     """Test Spark ranking evaluator ndcg."""
#     df_true, df_pred = spark_data
#
#     evaluator0 = Evaluator(df_true, df_true, top_k=10, col_prediction="rating")
#     assert evaluator0.ndcg_at_k() == 1.0
#
#     evaluator = Evaluator(df_true, df_pred, top_k=10)
#     assert evaluator.ndcg_at_k() == target_metrics['ndcg']
#
#     evaluator1 = Evaluator(df_true, df_pred,
#                            relevancy_method="by_threshold")
#     assert evaluator1.ndcg_at_k() == target_metrics['ndcg']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_map(spark_data, target_metrics):
#     """Test Spark ranking evaluator map."""
#     df_true, df_pred = spark_data
#
#     evaluator1 = Evaluator(top_k=10, rating_true=df_true, rating_pred=df_true,
#                            col_prediction="rating")
#     assert evaluator1.map_at_k() == 1.0
#
#     evaluator = Evaluator(df_true, df_pred, top_k=10)
#     assert evaluator.map_at_k() == target_metrics['map']
#
#     evaluator1 = Evaluator(
#         df_true, df_pred, relevancy_method="by_threshold")
#     assert evaluator1.map_at_k() == target_metrics['map']
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_get_metrics(spark_data, target_metrics):
#     """Test Spark get metrics."""
#     df_true, df_pred = spark_data
#
#     evaluator = Evaluator(df_true, df_pred)
#     assert RatingEvaluator.get_available_metrics() is not None
#     metrics = evaluator.get_metric()
#     assert metrics["rmse"] == target_metrics['rmse']
#
#     metrics = evaluator.get_metric(metrics="mae")
#     assert metrics["mae"] == target_metrics['mae']
#
#     metrics = evaluator.get_metric(_all=True)
#     assert metrics is not None
#
#     with pytest.raises(KeyError):
#         evaluator.get_metric(metrics="not_a_metric")
#
#     evaluator = RankingEvaluator(df_true, df_pred)
#     metrics = evaluator.get_metric()
#     assert metrics["precision@k"] == target_metrics['precision']
#
#     metrics = evaluator.get_metric(metrics="recall@k")
#     assert metrics["recall@k"] == target_metrics['recall']
#
#     metrics = evaluator.get_metric(_all=True)
#     assert metrics is not None
#
#     with pytest.raises(KeyError):
#         evaluator.get_metric(metrics="not_a_metric")
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_spark_errors(spark_data):
#     """Test Spark evaluator errors."""
#     df_true, df_pred = spark_data
#
#     with pytest.raises(ValueError):
#         RatingEvaluator(df_true, df_true, col_user="not_user")
#
#     with pytest.raises(ValueError):
#         RatingEvaluator(df_pred, df_pred,
#                         col_rating='prediction', col_user="not_user")
#
#     with pytest.raises(ValueError):
#         RatingEvaluator(df_true, df_pred, col_item="not_item")
#
#     with pytest.raises(ValueError):
#         RatingEvaluator(df_pred, df_pred,
#                         col_rating='prediction', col_item="not_item")
#
#     with pytest.raises(ValueError):
#         RatingEvaluator(df_true, df_pred, col_rating="not_rating")
#
#     with pytest.raises(ValueError):
#         RatingEvaluator(df_true, df_pred, col_prediction="not_prediction")
#
#     with pytest.raises(ValueError):
#         RankingEvaluator(df_true, df_true, col_user="not_user")
#
#     with pytest.raises(ValueError):
#         RankingEvaluator(df_pred, df_pred,
#                          col_rating='prediction', col_user="not_user")
#
#     with pytest.raises(ValueError):
#         RankingEvaluator(df_true, df_pred, col_item="not_item")
#
#     with pytest.raises(ValueError):
#         RankingEvaluator(df_pred, df_pred,
#                          col_rating='prediction', col_item="not_item")
#
#     with pytest.raises(ValueError):
#         RankingEvaluator(df_true, df_pred, col_rating="not_rating")
#
#     with pytest.raises(ValueError):
#         RankingEvaluator(df_true, df_pred, col_prediction="not_prediction")
#
#
# @pytest.mark.spark
# @pytest.mark.evaluation
# def test_evaluator_error():
#     with pytest.raises(NotImplementedError):
#         RatingEvaluator(str(''), str(''))
#     with pytest.raises(NotImplementedError):
#         RankingEvaluator(str(''), str(''))
