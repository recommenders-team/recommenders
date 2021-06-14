"""Tests for FBT methods."""
import pytest
import pandas as pd
from reco_utils.recommender.fbt.fbt import FBT
from pandas._testing import assert_frame_equal


@pytest.fixture(scope='module')
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


@pytest.fixture(scope='module')
def train_data(model_class):
    data_dict = {
            model_class.col_user: [1, 1, 1, 2, 2, 3],
            model_class.col_item: [11, 22, 33, 11, 33, 22],
            f'{model_class.col_item}_name': ['aa', 'bb', 'cc',
                                             'aa', 'cc', 'bb']
        }
    return pd.DataFrame(data_dict)


@pytest.fixture(scope='module')
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


# test dataframe for predict
@pytest.fixture(scope='module')
def test_data(model_class):
    data_dict = {
        model_class.col_user: [4, 4, 5, 5, 5],
        model_class.col_item: [33, 44, 11, 22, 33]
    }
    return pd.DataFrame(data_dict)


@pytest.fixture(scope='module')
def expected_predictions_test(model_class):
    data_dict = {
        model_class.col_user: [4, 4, 5, 5, 5],
        model_class.col_item: [11, 22, 11, 22, 33],
        model_class.col_score: [2.0, 1.0, 1.5, 1.0, 1.5]
    }
    return pd.DataFrame(data_dict)


def test_fbt_predict(model_class,
                     train_data,
                     test_data,
                     expected_predictions_test):
    """Test predict() method to work and fail as expected."""
    model_class.fit(train_data)
    # test output of predict function
    computed_predictions_test = model_class.predict(test_data)

    assert type(computed_predictions_test) == pd.DataFrame

    # test for expected output columns
    expected_column_names = {
        model_class.col_user,
        model_class.col_item,
        model_class.col_score
    }

    actual_column_names = set(computed_predictions_test.columns)
    assert expected_column_names == actual_column_names
    print(computed_predictions_test)

    # test the predict output on test_df
    assert_frame_equal(expected_predictions_test,
                       computed_predictions_test)


@pytest.fixture(scope='module')
def expected_k_predictions_test(model_class):
    data_dict = {
        model_class.col_user: [4, 5, 5],
        model_class.col_item: [11, 11, 33],
        model_class.col_score: [2.0, 1.5, 1.5]
    }
    return pd.DataFrame(data_dict)


# def test_fbt_init(self):
    #     """Test that FBT refuses to work with wrong inputs."""
    #     # incorrect input argument types should raise TypeErrors
    #     with self.assertRaises(TypeError):
    #         FBT(1, self.col_item, self.col_score, 1)
    #     with self.assertRaises(TypeError):
    #         FBT(self.col_user, 1, self.col_score, 1)
    #     with self.assertRaises(TypeError):
    #         FBT(self.col_user, self.col_item, 1, 1)
    #     with self.assertRaises(TypeError):
    #         FBT(self.col_user, self.col_item, self.col_score, 11)
    #     # missing arguments also won't work
    #     with self.assertRaises(TypeError):
    #         FBT()


#     def test_fbt_eval_map_at_k(self):
#         """Test evaluate() method to work and fail as expected."""
#         fbt = FBT(self.col_user, self.col_item, 1)
#         # bad input arguments should not work
#         with self.assertRaises(TypeError):
#             fbt.eval_map_at_k(self.test_df, self.non_df)

#         with self.assertRaises(TypeError):
#             fbt.eval_map_at_k(self.non_df,
#                               self.expected_predict_test_df)

#         # testing output of evaluate function
#         evaluate_output = fbt.eval_map_at_k(self.test_df,
#                                             self.expected_predict_test_df)
#         self.assertTrue(isinstance(evaluate_output, float))
#         self.assertAlmostEqual(evaluate_output, 0.5)

#         # test to make sure X and X_pred have same UserID's
#         arr1 = set(self.test_df[self.col_user].unique())
#         arr2 = set(self.expected_predict_test_df[self.col_user].unique())
#         self.assertEqual(arr1, arr2)
