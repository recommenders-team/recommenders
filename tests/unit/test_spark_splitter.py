# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pytest
from reco_utils.dataset.split_utils import min_rating_filter_spark
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

try:
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col
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
        Reference: https://stackoverflow.com/questions/41006182/generate-random-dates-within-a-range-in-numpy
        """
        days_to_add = np.arange(0, range_in_days)
        random_dates = []
        for i in range(range_in_days):
            random_date = np.datetime64(start_date) + np.random.choice(days_to_add)
            random_dates.append(random_date)

        return random_dates

    rating = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.randint(
                1, 5, test_specs["number_of_rows"]
            ),
            DEFAULT_ITEM_COL: np.random.randint(
                1, 15, test_specs["number_of_rows"]
            ),
            DEFAULT_RATING_COL: np.random.randint(
                1, 5, test_specs["number_of_rows"]
            ),
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01", test_specs["number_of_rows"]
            ),
        }
    )

    return rating


@pytest.fixture(scope="module")
def spark_dataset(python_data, spark):
    return spark.createDataFrame(python_data)


@pytest.mark.spark
def test_min_rating_filter(spark_dataset):
    dfs_user = min_rating_filter_spark(spark_dataset, min_rating=5, filter_by="user")
    dfs_item = min_rating_filter_spark(spark_dataset, min_rating=5, filter_by="item")

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
    splits = spark_random_split(
        spark_dataset, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["spark_randomsplit_tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["spark_randomsplit_tolerance"]
    )

    splits = spark_random_split(
        spark_dataset, ratio=test_specs["ratios"], seed=test_specs["seed"]
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
    splits = spark_chrono_split(
        spark_dataset, ratio=test_specs["ratio"], filter_by="user", min_rating=10
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

    assert _if_later(splits[0], splits[1])

    splits = spark_chrono_split(spark_dataset, ratio=test_specs["ratios"])

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert splits[2].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )

    assert _if_later(splits[0], splits[1])
    assert _if_later(splits[1], splits[2])


@pytest.mark.spark
def test_stratified_splitter(test_specs, spark_dataset):
    splits = spark_stratified_split(
        spark_dataset, ratio=test_specs["ratio"], filter_by="user", min_rating=10
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

    splits = spark_stratified_split(spark_dataset, ratio=test_specs["ratios"])

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
    dfs_rating = spark_dataset.withColumn(DEFAULT_TIMESTAMP_COL, col(DEFAULT_TIMESTAMP_COL).cast("float"))

    splits = spark_timestamp_split(
        dfs_rating, ratio=test_specs["ratio"], col_timestamp=DEFAULT_TIMESTAMP_COL
    )

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    max_split0 = splits[0].agg(F.max(DEFAULT_TIMESTAMP_COL)).first()[0]
    min_split1 = splits[1].agg(F.min(DEFAULT_TIMESTAMP_COL)).first()[0]
    assert(max_split0 <= min_split1)

    # Test multi split
    splits = spark_timestamp_split(dfs_rating, ratio=test_specs["ratios"])

    assert splits[0].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert splits[1].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert splits[2].count() / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )

    max_split0 = splits[0].agg(F.max(DEFAULT_TIMESTAMP_COL)).first()[0]
    min_split1 = splits[1].agg(F.min(DEFAULT_TIMESTAMP_COL)).first()[0]
    assert(max_split0 <= min_split1)

    max_split1 = splits[1].agg(F.max(DEFAULT_TIMESTAMP_COL)).first()[0]
    min_split2 = splits[2].agg(F.min(DEFAULT_TIMESTAMP_COL)).first()[0]
    assert(max_split1 <= min_split2)


def _if_later(data1, data2):
    """Helper function to test if records in data1 are earlier than that in data2.
    Returns:
        bool: True or False indicating if data1 is earlier than data2.
    """
    x = (data1.select(DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL)
         .groupBy(DEFAULT_USER_COL)
         .agg(F.max(DEFAULT_TIMESTAMP_COL).cast('long').alias('max'))
         .collect())
    max_times = {row[DEFAULT_USER_COL]: row['max'] for row in x}

    y = (data2.select(DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL)
         .groupBy(DEFAULT_USER_COL)
         .agg(F.min(DEFAULT_TIMESTAMP_COL).cast('long').alias('min'))
         .collect())
    min_times = {row[DEFAULT_USER_COL]: row['min'] for row in y}

    result = True
    for user, max_time in max_times.items():
        result = result and min_times[user] >= max_time

    return result
