# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
from itertools import product
import pytest

from reco_utils.dataset.split_utils import (
    min_rating_filter_pandas,
    split_pandas_data_with_ratios,
)

from reco_utils.dataset.python_splitters import (
    python_chrono_split,
    python_random_split,
    python_stratified_split,
)

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


@pytest.fixture(scope="module")
def test_specs():
    return {
        "number_of_rows": 1000,
        "user_ids": [1, 2, 3, 4, 5],
        "seed": 123,
        "ratio": 0.6,
        "ratios": [0.2, 0.3, 0.5],
        "split_numbers": [2, 3, 5],
        "tolerance": 0.01,
    }


@pytest.fixture(scope="module")
def python_dataset(test_specs):
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

    np.random.seed(test_specs["seed"])

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


def test_split_pandas_data(pandas_dummy_timestamp):
    splits = split_pandas_data_with_ratios(pandas_dummy_timestamp, ratios=[0.5, 0.5])
    assert len(splits[0]) == 5
    assert len(splits[1]) == 5

    splits = split_pandas_data_with_ratios(pandas_dummy_timestamp, ratios=[0.12, 0.36, 0.52])
    shape = pandas_dummy_timestamp.shape[0]
    assert len(splits[0]) == round(shape * 0.12)
    assert len(splits[1]) == round(shape * 0.36)
    assert len(splits[2]) == round(shape * 0.52)


def test_min_rating_filter(python_dataset):
    def count_filtered_rows(data, filter_by="user"):
        split_by_column = DEFAULT_USER_COL if filter_by == "user" else DEFAULT_ITEM_COL
        data_grouped = data.groupby(split_by_column)

        row_counts = []
        for name, group in data_grouped:
            data_group = data_grouped.get_group(name)
            row_counts.append(data_group.shape[0])

        return row_counts

    df_user = min_rating_filter_pandas(python_dataset, min_rating=5, filter_by="user")
    df_item = min_rating_filter_pandas(python_dataset, min_rating=5, filter_by="item")
    user_rating_counts = count_filtered_rows(df_user, filter_by="user")
    item_rating_counts = count_filtered_rows(df_item, filter_by="item")

    assert all(user_rating_counts)
    assert all(item_rating_counts)


def test_random_splitter(test_specs, python_dataset):
    """Test random splitter for Spark dataframes.

    NOTE: some split results may not match exactly with the ratios, which may be owing to the
    limited number of rows in
    the testing data. A approximate match with certain level of tolerance is therefore used
    instead for tests.
    """
    splits = python_random_split(
        python_dataset, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )
    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    splits = python_random_split(
        python_dataset, ratio=test_specs["ratios"], seed=test_specs["seed"]
    )

    assert len(splits) == 3
    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert len(splits[2]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )

    splits = python_random_split(
        python_dataset, ratio=test_specs["split_numbers"], seed=test_specs["seed"]
    )

    assert len(splits) == 3
    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert len(splits[2]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )


def test_chrono_splitter(test_specs, python_dataset):
    splits = python_chrono_split(
        python_dataset, ratio=test_specs["ratio"], min_rating=10, filter_by="user"
    )

    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # Test if both contains the same user list. This is because chrono split is stratified.
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)

    # Test all time stamps in test are later than that in train for all users.
    # This is for single-split case.
    max_train_times = splits[0][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).max()
    min_test_times = splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).min()
    check_times = max_train_times.join(min_test_times, lsuffix='_0', rsuffix='_1')
    assert all((check_times[DEFAULT_TIMESTAMP_COL + '_0'] < check_times[DEFAULT_TIMESTAMP_COL + '_1']).values)

    # Test multi-split case
    splits = python_chrono_split(
        python_dataset, ratio=test_specs["ratios"], min_rating=10, filter_by="user"
    )

    assert len(splits) == 3
    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert len(splits[2]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )

    # Test if all splits contain the same user list. This is because chrono split is stratified.
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    users_val = splits[2][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)
    assert set(users_train) == set(users_val)

    # Test if timestamps are correctly split. This is for multi-split case.
    max_train_times = splits[0][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).max()
    min_test_times = splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).min()
    check_times = max_train_times.join(min_test_times, lsuffix='_0', rsuffix='_1')
    assert all((check_times[DEFAULT_TIMESTAMP_COL + '_0'] < check_times[DEFAULT_TIMESTAMP_COL + '_1']).values)

    max_test_times = splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).max()
    min_val_times = splits[2][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]].groupby(DEFAULT_USER_COL).min()
    check_times = max_test_times.join(min_val_times, lsuffix='_1', rsuffix='_2')
    assert all((check_times[DEFAULT_TIMESTAMP_COL + '_1'] < check_times[DEFAULT_TIMESTAMP_COL + '_2']).values)


def test_stratified_splitter(test_specs, python_dataset):
    splits = python_stratified_split(
        python_dataset, ratio=test_specs["ratio"], min_rating=10, filter_by="user"
    )

    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # Test if both contains the same user list. This is because stratified split is stratified.
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()

    assert set(users_train) == set(users_test)

    splits = python_stratified_split(
        python_dataset, ratio=test_specs["ratios"], min_rating=10, filter_by="user"
    )

    assert len(splits) == 3
    assert len(splits[0]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][0], test_specs["tolerance"]
    )
    assert len(splits[1]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][1], test_specs["tolerance"]
    )
    assert len(splits[2]) / test_specs["number_of_rows"] == pytest.approx(
        test_specs["ratios"][2], test_specs["tolerance"]
    )
