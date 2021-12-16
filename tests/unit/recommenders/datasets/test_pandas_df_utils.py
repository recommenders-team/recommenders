# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from tempfile import TemporaryDirectory
import os

from recommenders.datasets.pandas_df_utils import (
    filter_by,
    LibffmConverter,
    has_same_base_dtype,
    has_columns,
    lru_cache_df,
    negative_feedback_sampler,
)


@pytest.fixture(scope="module")
def user_item_dataset():
    """Get users and items dataframe"""
    user_df = pd.DataFrame(
        {"user_id": [1, 2, 3, 4, 5], "user_age": [23, 24, 25, 26, 27]}
    )

    item_df = pd.DataFrame(
        {"item_id": [6, 7, 8], "item_feat": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]}
    )

    return user_df, item_df


def test_negative_feedback_sampler():
    df = pd.DataFrame(
        data={"userID": [1, 2, 3], "itemID": [1, 2, 3], "rating": [5, 5, 5]}
    )

    # Test ratio < 1
    sample_df = negative_feedback_sampler(
        df,
        col_user="userID",
        col_item="itemID",
        col_label="rating",
        ratio_neg_per_user=0.5,
    )
    assert sample_df.shape == (6, 3)
    assert sample_df.feedback.value_counts().to_dict() == {0: 3, 1: 3}
    for i in [1, 2, 3]:
        assert (
            sample_df[
                (sample_df.userID == i) & (sample_df.itemID == i)
            ].feedback.values[0]
            == 1
        )

    # Test ratio == 1
    sample_df = negative_feedback_sampler(
        df,
        col_user="userID",
        col_item="itemID",
        col_label="rating",
        ratio_neg_per_user=1,
    )
    assert sample_df.shape == (6, 3)
    assert sample_df.feedback.value_counts().to_dict() == {0: 3, 1: 3}
    for i in [1, 2, 3]:
        assert (
            sample_df[
                (sample_df.userID == i) & (sample_df.itemID == i)
            ].feedback.values[0]
            == 1
        )

    res_df = pd.DataFrame(
        data={
            "userID": [1, 2, 3, 1, 1, 2, 2, 3, 3],
            "itemID": [1, 2, 3, 2, 3, 1, 3, 1, 2],
            "feedback": [1, 1, 1, 0, 0, 0, 0, 0, 0],
        }
    )

    # Test ratio > 1
    sample_df = negative_feedback_sampler(
        df,
        col_user="userID",
        col_item="itemID",
        col_label="rating",
        ratio_neg_per_user=2,
    )
    assert sample_df.shape == (9, 3)
    assert sample_df.feedback.value_counts().to_dict() == {0: 6, 1: 3}
    assert np.all(
        sample_df.sort_values(["userID", "itemID"]).values
        == res_df.sort_values(["userID", "itemID"]).values
    )

    # Test too large ratio
    sample_df = negative_feedback_sampler(
        df,
        col_user="userID",
        col_item="itemID",
        col_label="rating",
        ratio_neg_per_user=3,
    )
    assert sample_df.shape == (9, 3)
    assert sample_df.feedback.value_counts().to_dict() == {0: 6, 1: 3}
    assert np.all(
        sample_df.sort_values(["userID", "itemID"]).values
        == res_df.sort_values(["userID", "itemID"]).values
    )

    # Test other options
    sample_df = negative_feedback_sampler(
        df,
        col_user="userID",
        col_item="itemID",
        col_label="rating",
        col_feedback="test_feedback",
        pos_value=2.4,
        neg_value=0.2,
        ratio_neg_per_user=3,
    )
    assert sample_df.columns[2] == "test_feedback"
    assert set(sample_df["test_feedback"].unique()) == set([2.4, 0.2])


def test_filter_by():
    user_df = pd.DataFrame(
        {"user_id": [1, 9, 3, 5, 5, 1], "item_id": [1, 6, 7, 6, 8, 9]}
    )

    seen_df = pd.DataFrame(
        {
            "user_id": [1, 2, 4],
        }
    )

    filtered_df = filter_by(user_df, seen_df, ["user_id"])

    # Check filtered out number
    assert len(filtered_df) == len(user_df) - 2
    # Check filtered out record
    assert len(filtered_df.loc[(user_df["user_id"] == 1)]) == 0


def test_csv_to_libffm():
    df_feature = pd.DataFrame(
        {
            "rating": [1, 0, 0, 1, 1],
            "field1": ["xxx1", "xxx2", "xxx4", "xxx4", "xxx4"],
            "field2": [3, 4, 5, 6, 7],
            "field3": [1.0, 2.0, 3.0, 4.0, 5.0],
            "field4": ["1", "2", "3", "4", "5"],
        }
    )

    with TemporaryDirectory() as td:
        filepath = os.path.join(td, "test")

        converter = LibffmConverter(filepath=filepath).fit(df_feature)
        df_feature_libffm = converter.transform(df_feature)

        # Check the input column types. For example, a bool type is not allowed.
        df_feature_wrong_type = df_feature.copy()
        df_feature_wrong_type["field4"] = True
        with pytest.raises(TypeError) as e:
            LibffmConverter().fit(df_feature_wrong_type)
            assert (
                e.value == "Input columns should be only object and/or numeric types."
            )

        # Check if the dim is the same.
        assert df_feature_libffm.shape == df_feature.shape

        # Check if the columns are converted successfully.
        assert df_feature_libffm.iloc[0, :].values.tolist() == [
            1,
            "1:1:1",
            "2:4:3",
            "3:5:1.0",
            "4:6:1",
        ]

        # Check if the duplicated column entries are indexed correctly.
        # It should skip counting the duplicated features in a field column.
        assert df_feature_libffm.iloc[-1, :].values.tolist() == [
            1,
            "1:3:1",
            "2:4:7",
            "3:5:5.0",
            "4:10:1",
        ]

        # Check if the file is written successfully.
        assert os.path.isfile(filepath)

        with open(filepath, "r") as f:
            line = f.readline()
            assert line == "1 1:1:1 2:4:3 3:5:1.0 4:6:1\n"

        # Parameters in the transformation should be reported correctly.
        params = converter.get_params()
        assert params == {"field count": 4, "feature count": 10, "file path": filepath}

        # Dataset with the same columns should be transformable with a fitted converter.
        df_feature_new = pd.DataFrame(
            {
                "rating": [1, 0, 0, 1, 1, 1],
                "field1": ["xxx1", "xxx2", "xxx4", "xxx4", "xxx4", "xxx3"],
                "field2": [3, 4, 5, 6, 7, 8],
                "field3": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "field4": ["1", "2", "3", "4", "5", "6"],
            }
        )
        df_feature_new_libffm = converter.transform(df_feature_new)

        assert df_feature_new_libffm.iloc[0, :].values.tolist() == [
            1,
            "1:1:1",
            "2:5:3",
            "3:6:1.0",
            "4:7:1",
        ]
        assert df_feature_new_libffm.iloc[-1, :].values.tolist() == [
            1,
            "1:4:1",
            "2:5:8",
            "3:6:6.0",
            "4:12:1",
        ]


def test_has_columns():
    df_1 = pd.DataFrame(dict(a=[1, 2, 3]))
    df_2 = pd.DataFrame(dict(b=[7, 8, 9], a=[1, 2, 3]))

    assert has_columns(df_1, ["a"])
    assert has_columns(df_2, ["a"])
    assert has_columns(df_2, ["a", "b"])
    assert not has_columns(df_2, ["a", "b", "c"])


def test_has_same_base_dtype():
    arr_int32 = np.array([1, 2, 3], dtype=np.int32)
    arr_int64 = np.array([1, 2, 3], dtype=np.int64)
    arr_float32 = np.array([1, 2, 3], dtype=np.float32)
    arr_float64 = np.array([1, 2, 3], dtype=np.float64)
    arr_str = ["a", "b", "c"]

    df_1 = pd.DataFrame(dict(a=arr_int32, b=arr_int64))
    df_2 = pd.DataFrame(dict(a=arr_int64, b=arr_int32))
    df_3 = pd.DataFrame(dict(a=arr_float32, b=arr_int32))
    df_4 = pd.DataFrame(dict(a=arr_float64, b=arr_float64))
    df_5 = pd.DataFrame(dict(a=arr_float64, b=arr_float64, c=arr_float64))
    df_6 = pd.DataFrame(dict(a=arr_str))

    # all columns match
    assert has_same_base_dtype(df_1, df_2)
    # specific column matches
    assert has_same_base_dtype(df_3, df_4, columns=["a"])
    # some column types do not match
    assert not has_same_base_dtype(df_3, df_4)
    # column types do not match
    assert not has_same_base_dtype(df_1, df_3, columns=["a"])
    # all columns are not shared
    assert not has_same_base_dtype(df_4, df_5)
    # column types do not match
    assert not has_same_base_dtype(df_5, df_6, columns=["a"])
    # assert string columns match
    assert has_same_base_dtype(df_6, df_6)


def test_lru_cache_df():
    df1 = pd.DataFrame(dict(a=[1, 2, 3], b=["a", "b", "c"]))
    df2 = pd.DataFrame(dict(a=[1, 2, 3], c=["a", "b", "c"]))
    df3 = pd.DataFrame(dict(a=[1, 2, 3], b=["a", "b", "d"]))

    @lru_cache_df(maxsize=2)
    def cached_func(df):
        pass

    assert "CacheInfo(hits=0, misses=0, maxsize=2, currsize=0)" == str(
        cached_func.cache_info()
    )
    cached_func(df1)
    assert "CacheInfo(hits=0, misses=1, maxsize=2, currsize=1)" == str(
        cached_func.cache_info()
    )
    cached_func(df1)
    assert "CacheInfo(hits=1, misses=1, maxsize=2, currsize=1)" == str(
        cached_func.cache_info()
    )
    cached_func(df2)
    assert "CacheInfo(hits=1, misses=2, maxsize=2, currsize=2)" == str(
        cached_func.cache_info()
    )
    cached_func(df2)
    assert "CacheInfo(hits=2, misses=2, maxsize=2, currsize=2)" == str(
        cached_func.cache_info()
    )
    cached_func(df3)
    assert "CacheInfo(hits=2, misses=3, maxsize=2, currsize=2)" == str(
        cached_func.cache_info()
    )
    cached_func(df1)
    assert "CacheInfo(hits=2, misses=4, maxsize=2, currsize=2)" == str(
        cached_func.cache_info()
    )
    cached_func(df3)
    assert "CacheInfo(hits=3, misses=4, maxsize=2, currsize=2)" == str(
        cached_func.cache_info()
    )
    cached_func.cache_clear()
    assert "CacheInfo(hits=0, misses=0, maxsize=2, currsize=0)" == str(
        cached_func.cache_info()
    )
