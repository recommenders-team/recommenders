# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from itertools import product
import pytest
from reco_utils.dataset.split_utils import min_rating_filter_spark
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

try:
    from reco_utils.common.spark_utils import start_or_get_spark
    from reco_utils.dataset.spark_splitters import (
        spark_chrono_split,
        spark_random_split,
        spark_stratified_split,
        spark_timestamp_split
    )
except ImportError:
    pass  # skip this import if we are in pure python environment


@pytest.fixture(scope="module")
def test_specs():
    return {
        "number_of_rows": 1000,
        "user_ids": [1, 2, 3, 4, 5],
        "seed": 1234,
        "ratio": 0.6,
        "ratios": [0.2, 0.3, 0.5],
        "tolerance": 0.05,
        "spark_randomsplit_tolerance": 0.1,
    }


@pytest.fixture(scope="module")
def python_data(test_specs):
    """Get Python labels"""

    def random_date_generator(start_date, range_in_days):
        """Helper function to generate random timestamps.

        Reference: https://stackoverflow.com/questions/41006182/generate-random-dates-within-a
        -range-in-numpy
        """
        days_to_add = np.arange(0, range_in_days)
        random_dates = []
        for i in range(range_in_days):
            random_date = np.datetime64(start_date) + np.random.choice(days_to_add)
            random_dates.append(random_date)

        return random_dates

    rating = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.random_integers(
                1, 5, test_specs["number_of_rows"]
            ),
            DEFAULT_ITEM_COL: np.random.random_integers(
                1, 15, test_specs["number_of_rows"]
            ),
            DEFAULT_RATING_COL: np.random.random_integers(
                1, 5, test_specs["number_of_rows"]
            ),
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01", test_specs["number_of_rows"]
            ),
        }
    )

    return rating


@pytest.fixture(scope="module")
def spark_dataset(python_data):
    """Get Python labels"""
    rating = python_data
    spark = start_or_get_spark("SplitterTesting")
    df_rating = spark.createDataFrame(rating)

    return df_rating


@pytest.mark.spark
def test_min_rating_filter(spark_dataset):
    """Test min rating filter
    """
    dfs_rating = spark_dataset

    dfs_user = min_rating_filter_spark(dfs_rating, min_rating=5, filter_by="user")
    dfs_item = min_rating_filter_spark(dfs_rating, min_rating=5, filter_by="item")

    user_rating_counts = [
        x["count"] >= 5 for x in dfs_user.groupBy(DEFAULT_USER_COL).count().collect()
    ]
    item_rating_counts = [
        x["count"] >= 5 for x in dfs_item.groupBy(DEFAULT_ITEM_COL).count().collect()
    ]

    assert all(user_rating_counts)
    assert all(item_rating_counts)


@pytest.mark.spark
def test_random_splitter(test_specs, spark_dataset):
    """Test random splitter for Spark dataframes.

    NOTE: some split results may not match exactly with the ratios, which may be owing to the
    limited number of rows in
    the testing data. A approximate match with certain level of tolerance is therefore used
    instead for tests.
    """
    df_rating = spark_dataset

    splits = spark_random_split(
        df_rating, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["spark_randomsplit_tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["spark_randomsplit_tolerance"]
    )

    splits = spark_random_split(
        df_rating, ratio=test_specs["ratios"], seed=test_specs["seed"]
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["spark_randomsplit_tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["spark_randomsplit_tolerance"]
    )
    assert splits[2].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["spark_randomsplit_tolerance"]
    )


@pytest.mark.spark
def test_chrono_splitter(test_specs, spark_dataset):
    """Test chronological splitter for Spark dataframes"""
    dfs_rating = spark_dataset

    splits = spark_chrono_split(
        dfs_rating, ratio=test_specs["ratio"], filter_by="user", min_rating=10
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # Test if both contains the same user list. This is because chrono split is stratified.
    users_train = (
        splits[0].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )
    users_test = (
        splits[1].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )

    assert set(users_train) == set(users_test)

    # Test all time stamps in test are later than that in train for all users.
    all_later = []
    for user in test_specs["user_ids"]:
        dfs_train = splits[0][splits[0][DEFAULT_USER_COL] == user]
        dfs_test = splits[1][splits[1][DEFAULT_USER_COL] == user]

        user_later = _if_later(dfs_train, dfs_test, col_timestamp=DEFAULT_TIMESTAMP_COL)

        all_later.append(user_later)
    assert all(all_later)

    splits = spark_chrono_split(dfs_rating, ratio=test_specs["ratios"])

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert splits[2].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )

    # Test if timestamps are correctly split. This is for multi-split case.
    all_later = []
    for user in test_specs["user_ids"]:
        dfs_train = splits[0][splits[0][DEFAULT_USER_COL] == user]
        dfs_valid = splits[1][splits[1][DEFAULT_USER_COL] == user]
        dfs_test = splits[2][splits[2][DEFAULT_USER_COL] == user]

        user_later_1 = _if_later(dfs_train, dfs_valid, col_timestamp=DEFAULT_TIMESTAMP_COL)
        user_later_2 = _if_later(dfs_valid, dfs_test, col_timestamp=DEFAULT_TIMESTAMP_COL)

        all_later.append(user_later_1)
        all_later.append(user_later_2)
    assert all(all_later)


@pytest.mark.spark
def test_stratified_splitter(test_specs, spark_dataset):
    """Test stratified splitter for Spark dataframes"""
    dfs_rating = spark_dataset

    splits = spark_stratified_split(
        dfs_rating, ratio=test_specs["ratio"], filter_by="user", min_rating=10
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # Test if both contains the same user list. This is because stratified split is stratified.
    users_train = (
        splits[0].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )
    users_test = (
        splits[1].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )

    assert set(users_train) == set(users_test)

    splits = spark_stratified_split(dfs_rating, ratio=test_specs["ratios"])

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert splits[2].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )


@pytest.mark.spark
def test_timestamp_splitter(test_specs, spark_dataset):
    """Test timestamp splitter for Spark dataframes"""
    from pyspark.sql.functions import col

    dfs_rating = spark_dataset
    dfs_rating = dfs_rating.withColumn(DEFAULT_TIMESTAMP_COL, col(DEFAULT_TIMESTAMP_COL).cast("float"))

    splits = spark_timestamp_split(
        dfs_rating, ratio=test_specs["ratio"], col_timestamp=DEFAULT_TIMESTAMP_COL
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # Test multi split
    splits = spark_stratified_split(dfs_rating, ratio=test_specs["ratios"])

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert splits[2].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )

    dfs_train = splits[0]
    dfs_valid = splits[1]
    dfs_test = splits[2]

    # if valid is later than train.
    all_later_1 = _if_later(dfs_train, dfs_valid, col_timestamp=DEFAULT_TIMESTAMP_COL)
    assert all_later_1

    # if test is later than valid.
    all_later_2 = _if_later(dfs_valid, dfs_test, col_timestamp=DEFAULT_TIMESTAMP_COL)
    assert all_later_2


def _if_later(data1, data2, col_timestamp=DEFAULT_TIMESTAMP_COL):
    '''Helper function to test if records in data1 are later than that in data2.

    Return:
        True or False indicating if data1 is later than data2.
    '''
    p = product(
        [
            x[col_timestamp]
            for x in data1.select(col_timestamp).collect()
        ],
        [
            x[col_timestamp]
            for x in data2.select(col_timestamp).collect()
        ],
    )

    if_late = [a <= b for (a, b) in p]

    return if_late

