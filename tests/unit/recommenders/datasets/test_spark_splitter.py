# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

try:
    from recommenders.datasets.split_utils import min_rating_filter_spark
    from pyspark.sql import functions as F
    from pyspark.sql.functions import col
    from recommenders.datasets.spark_splitters import (
        spark_chrono_split,
        spark_random_split,
        spark_stratified_split,
        spark_timestamp_split,
    )
except ImportError:
    pass  # skip this import if we are not in a spark environment


NUM_ROWS = 1000
RATIOS = [0.2, 0.3, 0.5]
SEED = 1234
TOL = 0.05


@pytest.fixture(scope="module")
def spark_dataset(spark):
    """Get spark dataframe"""

    return spark.createDataFrame(
        pd.DataFrame(
            {
                DEFAULT_USER_COL: np.random.randint(1, 5, NUM_ROWS),
                DEFAULT_ITEM_COL: np.random.randint(1, 15, NUM_ROWS),
                DEFAULT_RATING_COL: np.random.randint(1, 5, NUM_ROWS),
                DEFAULT_TIMESTAMP_COL: np.random.randint(1, 1000, NUM_ROWS)
                + np.datetime64("2018-01-01"),
            }
        )
    )


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
def test_random_splitter(spark_dataset):
    """Test random splitter for Spark dataframes.

    NOTE: some split results may not match exactly with the ratios, which may
    be owing to the limited number of rows in the testing data. A approximate
    match with certain level of tolerance is therefore used instead for tests."""

    splits = spark_random_split(spark_dataset, ratio=RATIOS[0], seed=SEED)

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], abs=TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(1 - RATIOS[0], abs=TOL)

    splits = spark_random_split(spark_dataset, ratio=RATIOS, seed=SEED)

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], abs=TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(RATIOS[1], abs=TOL)
    assert splits[2].count() / NUM_ROWS == pytest.approx(RATIOS[2], abs=TOL)


@pytest.mark.spark
def test_chrono_splitter(spark_dataset):
    splits = spark_chrono_split(
        spark_dataset, ratio=RATIOS[0], filter_by="user", min_rating=10
    )

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(1 - RATIOS[0], TOL)

    # Test if both contains the same user list. This is because chrono split is stratified.
    users_train = (
        splits[0].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )
    users_test = (
        splits[1].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )

    assert set(users_train) == set(users_test)

    assert _if_later(splits[0], splits[1])

    splits = spark_chrono_split(spark_dataset, ratio=RATIOS)

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(RATIOS[1], TOL)
    assert splits[2].count() / NUM_ROWS == pytest.approx(RATIOS[2], TOL)

    assert _if_later(splits[0], splits[1])
    assert _if_later(splits[1], splits[2])


@pytest.mark.spark
def test_stratified_splitter(spark_dataset):
    splits = spark_stratified_split(
        spark_dataset, ratio=RATIOS[0], filter_by="user", min_rating=10
    )

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(1 - RATIOS[0], TOL)

    # Test if there is intersection
    assert splits[0].intersect(splits[1]).count() == 0
    splits = spark_stratified_split(
        spark_dataset.repartition(4), ratio=RATIOS[0], filter_by="user", min_rating=10
    )
    assert splits[0].intersect(splits[1]).count() == 0

    # Test if both contains the same user list. This is because stratified split is stratified.
    users_train = (
        splits[0].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )
    users_test = (
        splits[1].select(DEFAULT_USER_COL).distinct().rdd.map(lambda r: r[0]).collect()
    )

    assert set(users_train) == set(users_test)

    splits = spark_stratified_split(spark_dataset, ratio=RATIOS)

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(RATIOS[1], TOL)
    assert splits[2].count() / NUM_ROWS == pytest.approx(RATIOS[2], TOL)

    # Test if there is intersection
    assert splits[0].intersect(splits[1]).count() == 0
    assert splits[0].intersect(splits[2]).count() == 0
    assert splits[1].intersect(splits[2]).count() == 0
    splits = spark_stratified_split(spark_dataset.repartition(9), ratio=RATIOS)
    assert splits[0].intersect(splits[1]).count() == 0
    assert splits[0].intersect(splits[2]).count() == 0
    assert splits[1].intersect(splits[2]).count() == 0


@pytest.mark.spark
def test_timestamp_splitter(spark_dataset):
    dfs_rating = spark_dataset.withColumn(
        DEFAULT_TIMESTAMP_COL, col(DEFAULT_TIMESTAMP_COL).cast("float")
    )

    splits = spark_timestamp_split(
        dfs_rating, ratio=RATIOS[0], col_timestamp=DEFAULT_TIMESTAMP_COL
    )

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(1 - RATIOS[0], TOL)

    max_split0 = splits[0].agg(F.max(DEFAULT_TIMESTAMP_COL)).first()[0]
    min_split1 = splits[1].agg(F.min(DEFAULT_TIMESTAMP_COL)).first()[0]
    assert max_split0 <= min_split1

    # Test multi split
    splits = spark_timestamp_split(dfs_rating, ratio=RATIOS)

    assert splits[0].count() / NUM_ROWS == pytest.approx(RATIOS[0], TOL)
    assert splits[1].count() / NUM_ROWS == pytest.approx(RATIOS[1], TOL)
    assert splits[2].count() / NUM_ROWS == pytest.approx(RATIOS[2], TOL)

    max_split0 = splits[0].agg(F.max(DEFAULT_TIMESTAMP_COL)).first()[0]
    min_split1 = splits[1].agg(F.min(DEFAULT_TIMESTAMP_COL)).first()[0]
    assert max_split0 <= min_split1

    max_split1 = splits[1].agg(F.max(DEFAULT_TIMESTAMP_COL)).first()[0]
    min_split2 = splits[2].agg(F.min(DEFAULT_TIMESTAMP_COL)).first()[0]
    assert max_split1 <= min_split2


def _if_later(data1, data2):
    """Helper function to test if records in data1 are earlier than that in data2.
    Returns:
        bool: True or False indicating if data1 is earlier than data2.
    """

    max_times = data1.groupBy(DEFAULT_USER_COL).agg(
        F.max(DEFAULT_TIMESTAMP_COL).alias("max")
    )
    min_times = data2.groupBy(DEFAULT_USER_COL).agg(
        F.min(DEFAULT_TIMESTAMP_COL).alias("min")
    )
    all_times = max_times.join(min_times, on=DEFAULT_USER_COL).select(
        (F.col("max") <= F.col("min"))
    )

    return all([x[0] for x in all_times.collect()])
