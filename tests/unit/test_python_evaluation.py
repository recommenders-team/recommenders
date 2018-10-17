"""
Test evaluation
"""
import numpy as np
import pandas as pd
import pytest

from utilities.evaluation.python_evaluation import PythonRatingEvaluation, PythonRankingEvaluation

TOL = 0.0001

@pytest.fixture
def target_metrics():
    return {
        'rmse': pytest.approx(7.254309, TOL),
        'mae': pytest.approx(6.375, TOL),
        'rsquared': pytest.approx(-31.699029, TOL),
        'exp_var': pytest.approx(-6.4466, 0.01),
        'ndcg': pytest.approx(0.38172, TOL),
        'precision': pytest.approx(0.26666, TOL),
        'map': pytest.approx(0.23613, TOL),
        'recall': pytest.approx(0.37777, TOL)
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


def test_init_python_rating_eval(python_data):
    """Test initializer python"""
    rating_true, rating_pred = python_data
    evaluator = PythonRatingEvaluation(rating_true, rating_pred)
    assert np.all(evaluator.rating_true == rating_true)
    assert np.all(evaluator.rating_pred == rating_pred)
    evaluator = PythonRankingEvaluation(rating_true, rating_pred)
    assert np.all(evaluator.rating_true == rating_true)
    assert np.all(evaluator.rating_pred == rating_pred)


def test_python_rmse(python_data, target_metrics):
    """Test Python evaluator RMSE"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.rmse() == 0
    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.rmse() == target_metrics['rmse']


def test_python_mae(python_data, target_metrics):
    """Test Python evaluator MAE"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.mae() == 0
    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.mae() == target_metrics['mae']


def test_python_rsquared(python_data, target_metrics):
    """Test Python evaluator rsquared"""
    rating_true, rating_pred = python_data

    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.rsquared() == pytest.approx(1.0, TOL)

    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.rsquared() == target_metrics['rsquared']


def test_python_exp_var(python_data, target_metrics):
    """Test Spark evaluator exp_var"""
    rating_true, rating_pred = python_data

    evaluator1 = PythonRatingEvaluation(rating_true=rating_true, rating_pred=rating_true,
                                        col_prediction="rating")
    assert evaluator1.exp_var() == pytest.approx(1.0, TOL)

    evaluator2 = PythonRatingEvaluation(rating_true, rating_pred)
    assert evaluator2.exp_var() == target_metrics['exp_var']


def test_python_ndcg_at_k(python_data, target_metrics):
    """Test Python evaluator NDCG"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.ndcg_at_k() == 1
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, 10)
    assert evaluator2.ndcg_at_k() == target_metrics['ndcg']


def test_python_map_at_k(python_data, target_metrics):
    """Test Python evaluator MAP"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.map_at_k() == 1
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, k=10)
    assert evaluator2.map_at_k() == target_metrics['map']


def test_python_precision(python_data, target_metrics):
    """Test Python evaluator precision"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.precision_at_k() == 0.6
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, k=10)
    assert evaluator2.precision_at_k() == target_metrics['precision']


def test_python_recall(python_data, target_metrics):
    """Test Python evaluator recall"""
    rating_true, rating_pred = python_data
    evaluator1 = PythonRankingEvaluation(k=10, rating_true=rating_true, rating_pred=rating_true,
                                         col_prediction="rating")
    assert evaluator1.recall_at_k() == pytest.approx(1, 0.1)
    evaluator2 = PythonRankingEvaluation(rating_true, rating_pred, k=10)
    assert evaluator2.recall_at_k() == target_metrics['recall']


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


