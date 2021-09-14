"""Tests for FBT methods."""
import pytest
import pandas as pd
from recommenders.models.fbt.fbt import FBT
from pandas._testing import assert_frame_equal
import numpy as np


@pytest.fixture
def model_class():
    return FBT(col_user="user_id",
               col_item="item_id",
               col_score="score",
               num_recos=1)


def test_init(model_class):
    assert model_class.col_user == "user_id"
    assert model_class.col_item == "item_id"
    assert model_class.col_score == "score"
    assert model_class.num_recos == 1


@pytest.fixture
def train_data(model_class):
    data_dict = {
            model_class.col_user: [1, 1, 1, 2, 2, 3],
            model_class.col_item: [11, 22, 33, 11, 33, 22],
            f'{model_class.col_item}_name': ['aa', 'bb', 'cc',
                                             'aa', 'cc', 'bb']
        }
    return pd.DataFrame(data_dict)


@pytest.fixture
def expected_fbt_model(model_class):
    model_df = pd.DataFrame({
            model_class.col_item: [11, 11, 22, 22, 33, 33],
            f'{model_class.col_item}_paired': [22, 33, 11, 33, 11, 22],
            model_class.col_score: [1, 2, 1, 1, 2, 1]
        })
    return model_df


def test_fbt_fit(model_class, train_data, expected_fbt_model):
    """Test fit() method to work and fail as expected."""
    # test output of fit function
    model_class.fit(train_data)
    computed_fbt_model = model_class._model_df.reset_index(drop=True)

    assert type(computed_fbt_model) == pd.DataFrame

    # test for expected output columns
    expected_column_names = {
        model_class.col_item,
        f'{model_class.col_item}_paired',
        model_class.col_score
    }
    actual_column_names = set(computed_fbt_model.columns)
    assert expected_column_names == actual_column_names

    # test the output dataframe on train_df
    assert_frame_equal(expected_fbt_model, computed_fbt_model)


# We will henceforth work with the trained model object
# instead of the untrained model_class object
@pytest.fixture
def model(train_data):
    fbt = FBT(col_user="user_id",
              col_item="item_id",
              col_score="score",
              num_recos=1)

    fbt.fit(train_data)
    return fbt


# test dataframe for predict
@pytest.fixture
def test_data(model):
    data_dict = {
        model.col_user: [1, 1, 2, 2, 2],
        model.col_item: [33, 44, 11, 22, 33]
    }
    return pd.DataFrame(data_dict)


@pytest.fixture
def expected_predictions_test(model):
    data_dict = {
        model.col_user: [1, 1, 2, 2, 2],
        model.col_item: [11, 22, 11, 22, 33],
        model.col_score: [2.0, 1.0, 1.5, 1.0, 1.5]
    }
    return pd.DataFrame(data_dict)


def test_fbt_predict(model,
                     test_data,
                     expected_predictions_test):
    """Test predict() method to work and fail as expected."""
    # test output of predict function
    computed_predictions_test = model.predict(test_data)

    assert type(computed_predictions_test) == pd.DataFrame

    # test for expected output columns
    expected_column_names = {
        model.col_user,
        model.col_item,
        model.col_score
    }

    actual_column_names = set(computed_predictions_test.columns)
    assert expected_column_names == actual_column_names

    # test the predict output on test_df
    assert_frame_equal(expected_predictions_test,
                       computed_predictions_test)


@pytest.fixture
def test_k_preds(model):
    data_dict = {
        model.col_user: [1, 2],
        model.col_item: [11, 11],
        model.col_score: [2.0, 1.5],
        'rank': [1, 1]
    }
    test_k_preds.seen_false = pd.DataFrame(data_dict)

    # TODO: need to fix this example
    data_dict = {
        model.col_user: [1, 2],
        model.col_item: [None, 22.0],
        model.col_score: [None, 1.0],
        'rank': [None, 1.0]
    }
    test_k_preds.seen_true = pd.DataFrame(data_dict)
    return test_k_preds


@pytest.mark.parametrize("remove_seen, train", [False, True])
def test_recommend_k_items(model,
                           train,
                           test_data,
                           test_k_preds,
                           remove_seen):

    train = None
    if remove_seen:
        train = train_data

    # test output of predict function
    computed_k_preds = (
        model
        .recommend_k_items(test=test_data,
                           train=train,
                           remove_seen=remove_seen,
                           top_k=1)
    )

    assert type(computed_k_preds) == pd.DataFrame

    # test for expected output columns
    expected_column_names = {
        model.col_user,
        model.col_item,
        model.col_score,
        'rank'
    }

    actual_column_names = set(computed_k_preds.columns)
    assert expected_column_names == actual_column_names

    # test the predict output on test_df
    if remove_seen:
        assert_frame_equal(test_k_preds.seen_true,
                           computed_k_preds)
    else:
        assert_frame_equal(test_k_preds.seen_false,
                           computed_k_preds)


def test_fbt_eval_map_at_k(model,
                           test_data,
                           expected_predictions_test):
    """Test evaluate() method to work and fail as expected."""
    # testing output of evaluate function
    evaluate_output = model.eval_map_at_k(test_data,
                                          expected_predictions_test)
    assert type(evaluate_output) == np.float64
    assert evaluate_output == pytest.approx(0.5)

    # test to make sure X and X_pred have same UserID's
    arr1 = set(test_data[model.col_user].unique())
    arr2 = set(expected_predictions_test[model.col_user].unique())
    assert arr1 == arr2
