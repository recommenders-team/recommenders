# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import pandas as pd
from reco_utils.dataset.pandas_df_utils import (
    user_item_pairs,
    filter_by,
    libffm_converter,
    negative_feedback_sampler
)


@pytest.fixture(scope="module")
def user_item_dataset():
    """Get users and items dataframe"""
    user_df = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'user_age': [23, 24, 25, 26, 27]
    })

    item_df = pd.DataFrame({
        'item_id': [6, 7, 8],
        'item_feat': [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]
    })

    return user_df, item_df


def test_user_item_pairs(user_item_dataset):
    user_df, item_df = user_item_dataset

    user_item = user_item_pairs(
        user_df=user_df,
        item_df=item_df,
        user_col='user_id',
        item_col='item_id',
        shuffle=False
    )
    # Validate cross-join
    assert len(user_df) * len(item_df) == len(user_item)
    assert user_item.loc[(user_item['user_id'] == 3) & (user_item['item_id'] == 7)].values.tolist()[0]\
        == [3, 25, 7, [0.2, 0.2]]

    # Check if result is deterministic
    assert user_item.iloc[0].values.tolist() == [1, 23, 6, [0.1, 0.1]]

    # Check shuffle
    user_item_shuffled = user_item_pairs(
        user_df=user_df,
        item_df=item_df,
        user_col='user_id',
        item_col='item_id',
        shuffle=True
    )
    # Check shuffled result is still valid
    assert len(user_df) * len(item_df) == len(user_item_shuffled)
    row = user_item.loc[(user_item['user_id'] == 2) & (user_item['item_id'] == 6)]
    assert row['user_age'].iloc[0] == 24
    assert row['item_feat'].iloc[0] == [0.1, 0.1]
    # Check shuffled result is different from not-shuffled dataframe
    assert [*user_item_shuffled['user_id'].values] != [*user_item['user_id'].values]

    # Check filter
    seen_df = pd.DataFrame({
        'user_id': [1, 9, 3, 5, 5, 1],
        'item_id': [1, 6, 7, 6, 8, 9]
    })
    user_item_filtered = user_item_pairs(
        user_df=user_df,
        item_df=item_df,
        user_col='user_id',
        item_col='item_id',
        user_item_filter_df=seen_df,
        shuffle=False
    )
    # Check filtered out number
    assert len(user_item_filtered) == len(user_item) - 3
    # Check filtered out record
    assert len(user_item_filtered.loc[(user_item['user_id'] == 3) & (user_item['item_id'] == 7)]) == 0


def test_filter_by():
    user_df = pd.DataFrame({
        'user_id': [1, 9, 3, 5, 5, 1],
        'item_id': [1, 6, 7, 6, 8, 9]
    })

    seen_df = pd.DataFrame({
        'user_id': [1, 2, 4],
    })

    filtered_df = filter_by(user_df, seen_df, ['user_id'])

    # Check filtered out number
    assert len(filtered_df) == len(user_df) - 2
    # Check filtered out record
    assert len(filtered_df.loc[(user_df['user_id'] == 1)]) == 0


def test_csv_to_libffm():
    df_feature = pd.DataFrame({
        'rating': [1, 0, 0, 1, 1],
        'field1': ['xxx1', 'xxx2', 'xxx4', 'xxx4', 'xxx4'],
        'field2': [3, 4, 5, 6, 7],
        'field3': [1.0, 2.0, 3.0, 4.0, 5.0],
        'field4': ['1', '2', '3', '4', '5']
    })

    import tempfile
    import os

    filedir = tempfile.tempdir
    filename = 'test'
    filepath = os.path.join(filedir, filename)

    # Check the input column types. For example, a bool type is not allowed.
    df_feature_wrong_type = df_feature.copy()
    df_feature_wrong_type['field4'] = True
    with pytest.raises(TypeError) as e:
        libffm_converter(df_feature_wrong_type, col_rating='rating')
        assert e.value == "Input columns should be only object and/or numeric types."

    df_feature_libffm = libffm_converter(df_feature, col_rating='rating', filepath=filepath)

    # Check if the dim is the same.
    assert df_feature_libffm.shape == df_feature.shape

    # Check if the columns are converted successfully.
    assert df_feature_libffm.iloc[0, :].values.tolist() == [1, '1:1:1', '2:2:3', '3:3:1.0', '4:4:1']

    # Check if the duplicated column entries are indexed correctly.
    # It should skip counting the duplicated features in a field column.
    assert df_feature_libffm.iloc[-1, :].values.tolist() == [1, '1:3:1', '2:2:7', '3:3:5.0', '4:8:1']

    # Check if the file is written successfully.
    assert os.path.isfile(filepath)

    with open(filepath, 'r') as f:
        line = f.readline()
        assert line == '1 1:1:1 2:2:3 3:3:1.0 4:4:1\n'


def test_negative_feedback_sampler():
    df = pd.DataFrame({
        'userID': [1, 2, 3, 4, 4, 5, 5, 5],
        'itemID': [1, 2, 3, 1, 2, 1, 2, 3],
        'rating': [5, 5, 5, 5, 5, 5, 5, 5]
    })

    # Same amount of negative samples to the positive samples per user
    df_neg_sampled_1 = negative_feedback_sampler(
        df, 
        col_user='userID', 
        col_item='itemID', 
        col_label='label', 
        ratio_neg_per_user=1
    )

    # Other than user #4 and #5, all the other users should have the same number of positive and negative feedback
    assert (
        df_neg_sampled_1[(df_neg_sampled_1['label'] == 0) & (df_neg_sampled_1['userID'].isin([1, 2, 3]))].shape[0]
        == df_neg_sampled_1[(df_neg_sampled_1['label'] == 1) & (df_neg_sampled_1['userID'].isin([1, 2, 3]))].shape[0]
    )

    # For user #4, the negative feedback samples should be the number of the min of total positive feedback and the
    # total possible number of negative feedback samples (in our case, it is 1).
    assert (
        df_neg_sampled_1[(df_neg_sampled_1['label'] == 0) & (df_neg_sampled_1['userID'] == 4)].shape[0]
        == 1
    )

    # For user #5, the negative feedback samples should be 0 as he has interacted with all the items.
    assert (
        df_neg_sampled_1[(df_neg_sampled_1['label'] == 0) & (df_neg_sampled_1['userID'] == 5)].shape[0]
        == 0
    )

    # Label column specified by the user should be there in the output.
    columns_new = df_neg_sampled_1.columns
    assert 'label' in columns_new

    # If there is no 'rating' column, it should still work.
    df_neg_sampled_11 = negative_feedback_sampler(
        df.drop('rating', axis=1), 
        col_user='userID', 
        col_item='itemID', 
        col_label='label',
        ratio_neg_per_user=1
    )
    assert (
        df_neg_sampled_1[df_neg_sampled_1['label'] == 0 & df_neg_sampled_1['userID'].isin([1, 2, 3])].shape[0]
        == df_neg_sampled_1[df_neg_sampled_1['label'] == 1 & df_neg_sampled_1['userID'].isin([1, 2, 3])].shape[0]
    )

    # If the ratio is different, say 2, it should still work.
    df_neg_sampled_2 = negative_feedback_sampler(
        df, 
        col_user='userID', 
        col_item='itemID', 
        col_label='label', 
        ratio_neg_per_user=2
    )
    assert (
        df_neg_sampled_2[(df_neg_sampled_2['label'] == 0) & (df_neg_sampled_2['userID'].isin([1, 2, 3]))].shape[0]
        == 6
    )
    assert (
        df_neg_sampled_2[(df_neg_sampled_2['label'] == 0) & (df_neg_sampled_2['userID'] == 4)].shape[0]
        == 1
    )
    assert (
        df_neg_sampled_2[(df_neg_sampled_2['label'] == 0) & (df_neg_sampled_2['userID'] == 5)].shape[0]
        == 0
    )

    # If the ratio is smaller than 1, it should at least sample one negative feedback (if there exist more than one
    # negative feedback for the user).
    df_neg_sampled_3 = negative_feedback_sampler(
        df, 
        col_user='userID', 
        col_item='itemID', 
        col_label='label', 
        ratio_neg_per_user=0.5
    )
    assert (
        df_neg_sampled_3[(df_neg_sampled_3['label'] == 0) & (df_neg_sampled_3['userID'].isin([1, 2, 3]))].shape[0]
        == 3
    )
    assert (
        df_neg_sampled_3[(df_neg_sampled_3['label'] == 0) & (df_neg_sampled_3['userID'] == 4)].shape[0]
        == 1
    )
    assert (
        df_neg_sampled_3[(df_neg_sampled_3['label'] == 0) & (df_neg_sampled_3['userID'] == 5)].shape[0]
        == 0
    )