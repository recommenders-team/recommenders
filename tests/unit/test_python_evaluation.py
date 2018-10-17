"""
Test evaluation
"""
import numpy as np
import pandas as pd
import pytest

from utilities.evaluation.python_evaluation import PythonRatingEvaluation, PythonRankingEvaluation
from utilities.evaluation.spark_evaluation import SparkRankingEvaluation, SparkRatingEvaluation
from utilities.common.spark_utils import start_or_get_spark


@pytest.fixture
def target_metrics():
    return {
        'rmse': pytest.approx(7.254309, 0.0001),
        'mae': pytest.approx(6.375, 0.0001),
        'rsquared': pytest.approx(-31.699029, 0.0001),
        'exp_var': pytest.approx(-6.4466, 0.01),
        'ndcg': pytest.approx(0.38172, 0.0001),
        'precision': pytest.approx(0.26666, 0.0001),
        'map': pytest.approx(0.23613, 0.0001),
        'recall': pytest.approx(0.37777, 0.0001)
    }


@pytest.fixture(scope='module')
def python_data():
    """Get Python labels"""
    rating_true = pd.DataFrame({
        'userID': [1, 1, 1,
                   2, 2, 2, 2, 2,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'itemID': [1, 2, 3,
                   1, 4, 5, 6, 7,
                   2, 5, 6, 8, 9, 10, 11, 12, 13, 14],
        'rating': [5, 4, 3,
                   5, 5, 3, 3, 1,
                   5, 5, 5, 4, 4, 3, 3, 3, 2, 1]
    })
    rating_pred = pd.DataFrame({
        'userID': [1, 1, 1,
                   2, 2, 2, 2, 2,
                   3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        'itemID': [3, 10, 12,
                   10, 3, 5, 11, 13,
                   4, 10, 7, 13, 1, 3, 5, 2, 11, 14
                   ],
        'prediction': [14, 13, 12,
                       14, 13, 12, 11, 10,
                       14, 13, 12, 11, 10, 9, 8, 7, 6, 5]
    })
    return rating_true, rating_pred


@pytest.mark.evaluation
def test_init_python_rating_eval(python_data):
    """Test initializer python"""
    rating_true, rating_pred = python_data
    evaluator = PythonRatingEvaluation(rating_true, rating_pred)
    assert np.all(evaluator.rating_true == rating_true)
    assert np.all(evaluator.rating_pred == rating_pred)
    evaluator = PythonRankingEvaluation(rating_true, rating_pred)
    assert np.all(evaluator.rating_true == rating_true)
    assert np.all(evaluator.rating_pred == rating_pred)


@pytest.mark.evaluation
def test_python_rmse(python_data, target_metrics):
    """Test Python evaluator RMSE"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.rmse() == 0
    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.rmse() == target_metrics['rmse']


@pytest.mark.evaluation
def test_python_mae(python_data, target_metrics):
    """Test Python evaluator MAE"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.mae() == 0
    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.mae() == target_metrics['mae']


@pytest.mark.evaluation
def test_python_rsquared(python_data, target_metrics):
    """Test Python evaluator rsquared"""
    rating_true, rating_pred = python_data

    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.rsquared() == pytest.approx(1.0, 0.0001)

    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.rsquared() == target_metrics['rsquared']


@pytest.mark.evaluation
def test_python_exp_var(python_data, target_metrics):
    """Test Spark evaluator exp_var"""
    rating_true, rating_pred = python_data

    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.exp_var() == pytest.approx(1.0, 0.0001)

    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.exp_var() == target_metrics['exp_var']


@pytest.mark.evaluation
def test_python_ndcg_at_k(python_data, target_metrics):
    """Test Python evaluator NDCG"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.ndcg_at_k() == 1
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, 10)
    assert evaluator2.ndcg_at_k() == target_metrics['ndcg']


@pytest.mark.evaluation
def test_python_map_at_k(python_data, target_metrics):
    """Test Python evaluator MAP"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.map_at_k() == 1
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, k=10)
    assert evaluator2.map_at_k() == target_metrics['map']


@pytest.mark.evaluation
def test_python_precision(python_data, target_metrics):
    """Test Python evaluator precision"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.precision_at_k() == 0.6
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, k=10)
    assert evaluator2.precision_at_k() == target_metrics['precision']


@pytest.mark.evaluation
def test_python_recall(python_data, target_metrics):
    """Test Python evaluator recall"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.recall_at_k() == pytest.approx(1, 0.1)
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, k=10)
    assert evaluator2.recall_at_k() == target_metrics['recall']


@pytest.mark.evaluation
def test_python_errors(python_data):
    """Test Python evaluator errors."""
    rating_true, rating_pred = python_data

    with pytest.raises(ValueError):
        PythonRatingEvaluation(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        PythonRatingEvaluation(rating_pred, rating_pred,
                               col_rating='prediction', col_user="not_user")

    with pytest.raises(ValueError):
        PythonRatingEvaluation(rating_true, rating_pred, col_item="not_item")

    with pytest.raises(ValueError):
        PythonRatingEvaluation(rating_pred, rating_pred,
                               col_rating='prediction', col_item="not_item")

    with pytest.raises(ValueError):
        PythonRatingEvaluation(rating_true, rating_pred, col_rating="not_rating")

    with pytest.raises(ValueError):
        PythonRatingEvaluation(rating_true, rating_pred, col_prediction="not_prediction")

    with pytest.raises(ValueError):
        PythonRankingEvaluation(rating_true, rating_true, col_user="not_user")

    with pytest.raises(ValueError):
        PythonRankingEvaluation(rating_pred, rating_pred,
                                col_rating='prediction', col_user="not_user")

    with pytest.raises(ValueError):
        PythonRankingEvaluation(rating_true, rating_pred, col_item="not_item")

    with pytest.raises(ValueError):
        PythonRankingEvaluation(rating_pred, rating_pred,
                                col_rating='prediction', col_item="not_item")

    with pytest.raises(ValueError):
        PythonRankingEvaluation(rating_true, rating_pred, col_rating="not_rating")

    with pytest.raises(ValueError):
        PythonRankingEvaluation(rating_true, rating_pred, col_prediction="not_prediction")


@pytest.mark.spark
def test_spark_python_match(python_data):
    # Test on the original data with k = 10.

    df_true, df_pred = python_data

    eval_python1 = PythonRankingEvaluation(df_true, df_pred, k=10)

    spark = start_or_get_spark()
    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark1 = SparkRankingEvaluation(dfs_true, dfs_pred, k=10)

    match1 = [
        eval_python1.recall_at_k() == pytest.approx(eval_spark1.recall_at_k(), 0.0001),
        eval_python1.precision_at_k() == pytest.approx(eval_spark1.precision_at_k(), 0.0001),
        eval_python1.ndcg_at_k() == pytest.approx(eval_spark1.ndcg_at_k(), 0.0001),
        eval_python1.map_at_k() == pytest.approx(eval_spark1.map_at_k(), 0.0001)
    ]

    assert all(match1)

    # Test on the original data with k = 3.

    eval_python2 = PythonRankingEvaluation(df_true, df_pred, k=3)

    spark = start_or_get_spark()
    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark2 = SparkRankingEvaluation(dfs_true, dfs_pred, k=3)

    match2 = [
        eval_python2.recall_at_k() == pytest.approx(eval_spark2.recall_at_k(), 0.0001),
        eval_python2.precision_at_k() == pytest.approx(eval_spark2.precision_at_k(), 0.0001),
        eval_python2.ndcg_at_k() == pytest.approx(eval_spark2.ndcg_at_k(), 0.0001),
        eval_python2.map_at_k() == pytest.approx(eval_spark2.map_at_k(), 0.0001)
    ]

    assert all(match2)

    # Remove the first row from the original data.

    df_pred = df_pred[1:-1]

    eval_python3 = PythonRankingEvaluation(df_true, df_pred, k=10)

    spark = start_or_get_spark()
    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark3 = SparkRankingEvaluation(dfs_true, dfs_pred, k=10)

    match3 = [
        eval_python3.recall_at_k() == pytest.approx(eval_spark3.recall_at_k(), 0.0001),
        eval_python3.precision_at_k() == pytest.approx(eval_spark3.precision_at_k(), 0.0001),
        eval_python3.ndcg_at_k() == pytest.approx(eval_spark3.ndcg_at_k(), 0.0001),
        eval_python3.map_at_k() == pytest.approx(eval_spark3.map_at_k(), 0.0001)
    ]

    assert all(match3)

    # Test with one user

    df_pred = df_pred[df_pred["userID"] == 3]
    df_true = df_true[df_true["userID"] == 3]

    eval_python4 = PythonRankingEvaluation(df_true, df_pred, k=10)

    spark = start_or_get_spark()
    dfs_true = spark.createDataFrame(df_true)
    dfs_pred = spark.createDataFrame(df_pred)

    eval_spark4 = SparkRankingEvaluation(dfs_true, dfs_pred, k=10)

    match4 = [
        eval_python4.recall_at_k() == pytest.approx(eval_spark4.recall_at_k(), 0.0001),
        eval_python4.precision_at_k() == pytest.approx(eval_spark4.precision_at_k(), 0.0001),
        eval_python4.ndcg_at_k() == pytest.approx(eval_spark4.ndcg_at_k(), 0.0001),
        eval_python4.map_at_k() == pytest.approx(eval_spark4.map_at_k(), 0.0001)
    ]

    assert all(match4)

