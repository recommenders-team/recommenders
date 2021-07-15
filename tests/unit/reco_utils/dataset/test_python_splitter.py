# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pytest

from reco_utils.dataset.split_utils import (
    min_rating_filter_pandas,
    split_pandas_data_with_ratios,
)

from reco_utils.dataset.python_splitters import (
    python_chrono_split,
    python_random_split,
    python_stratified_split,
    numpy_stratified_split,
)

from reco_utils.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


@pytest.fixture(scope="module")
def test_specs():
    return {
        "number_of_rows": 1000,
        "seed": 123,
        "ratio": 0.6,
        "ratios": [0.2, 0.3, 0.5],
        "split_numbers": [2, 3, 5],
        "tolerance": 0.01,
        "number_of_items": 50,
        "number_of_users": 20,
        "fluctuation": 0.02,
    }


@pytest.fixture(scope="module")
def python_dataset(test_specs):
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

    np.random.seed(test_specs["seed"])

    rating = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.randint(1, 5, test_specs["number_of_rows"]),
            DEFAULT_ITEM_COL: np.random.randint(1, 15, test_specs["number_of_rows"]),
            DEFAULT_RATING_COL: np.random.randint(1, 6, test_specs["number_of_rows"]),
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01", test_specs["number_of_rows"]
            ),
        }
    )
    return rating


@pytest.fixture(scope="module")
def python_int_dataset(test_specs):
    np.random.seed(test_specs["seed"])

    # generates the user/item affinity matrix. Ratings are in the interval [0, 5), with 0s denoting unrated items
    return np.random.randint(
        low=0,
        high=6,
        size=(test_specs["number_of_users"], test_specs["number_of_items"]),
    )


@pytest.fixture(scope="module")
def python_float_dataset(test_specs):
    np.random.seed(test_specs["seed"])

    # generates the user/item affinity matrix. Ratings are in the interval [0, 5), with 0s denoting unrated items.
    return (
        np.random.random(
            size=(test_specs["number_of_users"], test_specs["number_of_items"])
        )
        * 5
    )


def test_split_pandas_data(pandas_dummy_timestamp):
    splits = split_pandas_data_with_ratios(pandas_dummy_timestamp, ratios=[0.5, 0.5])
    assert len(splits[0]) == 5
    assert len(splits[1]) == 5

    splits = split_pandas_data_with_ratios(
        pandas_dummy_timestamp, ratios=[0.12, 0.36, 0.52]
    )
    shape = pandas_dummy_timestamp.shape[0]
    assert len(splits[0]) == round(shape * 0.12)
    assert len(splits[1]) == round(shape * 0.36)
    assert len(splits[2]) == round(shape * 0.52)

    with pytest.raises(ValueError):
        splits = split_pandas_data_with_ratios(
            pandas_dummy_timestamp, ratios=[0.6, 0.2, 0.4]
        )


def test_min_rating_filter():
    python_dataset = pd.DataFrame(
        {
            DEFAULT_USER_COL: [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5],
            DEFAULT_ITEM_COL: [5, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1],
            DEFAULT_RATING_COL: np.random.randint(1, 6, 15),
        }
    )

    def count_filtered_rows(data, filter_by="user"):
        split_by_column = DEFAULT_USER_COL if filter_by == "user" else DEFAULT_ITEM_COL
        data_grouped = data.groupby(split_by_column)

        row_counts = []
        for name, group in data_grouped:
            data_group = data_grouped.get_group(name)
            row_counts.append(data_group.shape[0])

        return row_counts

    df_user = min_rating_filter_pandas(python_dataset, min_rating=3, filter_by="user")
    df_item = min_rating_filter_pandas(python_dataset, min_rating=2, filter_by="item")
    user_rating_counts = count_filtered_rows(df_user, filter_by="user")
    item_rating_counts = count_filtered_rows(df_item, filter_by="item")

    assert all(u >= 3 for u in user_rating_counts)
    assert all(i >= 2 for i in item_rating_counts)


def test_random_splitter(test_specs, python_dataset):
    """NOTE: some split results may not match exactly with the ratios, which may be owing to the  limited number of
    rows in the testing data. A approximate match with certain level of tolerance is therefore used instead for tests.
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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

    # check values sum to 1
    splits = python_random_split(
        python_dataset, ratio=[0.7, 0.2, 0.1], seed=test_specs["seed"]
    )

    assert (len(splits)) == 3


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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

    # Test if both contains the same user list. This is because chrono split is stratified.
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)

    # Test all time stamps in test are later than that in train for all users.
    # This is for single-split case.
    max_train_times = (
        splits[0][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]]
        .groupby(DEFAULT_USER_COL)
        .max()
    )
    min_test_times = (
        splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]]
        .groupby(DEFAULT_USER_COL)
        .min()
    )
    check_times = max_train_times.join(min_test_times, lsuffix="_0", rsuffix="_1")
    assert all(
        (
            check_times[DEFAULT_TIMESTAMP_COL + "_0"]
            < check_times[DEFAULT_TIMESTAMP_COL + "_1"]
        ).values
    )

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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

    # Test if all splits contain the same user list. This is because chrono split is stratified.
    users_train = splits[0][DEFAULT_USER_COL].unique()
    users_test = splits[1][DEFAULT_USER_COL].unique()
    users_val = splits[2][DEFAULT_USER_COL].unique()
    assert set(users_train) == set(users_test)
    assert set(users_train) == set(users_val)

    # Test if timestamps are correctly split. This is for multi-split case.
    max_train_times = (
        splits[0][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]]
        .groupby(DEFAULT_USER_COL)
        .max()
    )
    min_test_times = (
        splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]]
        .groupby(DEFAULT_USER_COL)
        .min()
    )
    check_times = max_train_times.join(min_test_times, lsuffix="_0", rsuffix="_1")
    assert all(
        (
            check_times[DEFAULT_TIMESTAMP_COL + "_0"]
            < check_times[DEFAULT_TIMESTAMP_COL + "_1"]
        ).values
    )

    max_test_times = (
        splits[1][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]]
        .groupby(DEFAULT_USER_COL)
        .max()
    )
    min_val_times = (
        splits[2][[DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL]]
        .groupby(DEFAULT_USER_COL)
        .min()
    )
    check_times = max_test_times.join(min_val_times, lsuffix="_1", rsuffix="_2")
    assert all(
        (
            check_times[DEFAULT_TIMESTAMP_COL + "_1"]
            < check_times[DEFAULT_TIMESTAMP_COL + "_2"]
        ).values
    )


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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)

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

    for split in splits:
        assert set(split.columns) == set(python_dataset.columns)


def test_int_numpy_stratified_splitter(test_specs, python_int_dataset):
    # generate a syntetic dataset
    X = python_int_dataset

    # the splitter returns (in order): train and test user/affinity matrices, train and test datafarmes and user/items to matrix maps
    Xtr, Xtst = numpy_stratified_split(
        X, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    # check that the generated matrices have the correct dimensions
    assert (Xtr.shape[0] == X.shape[0]) & (Xtr.shape[1] == X.shape[1])
    assert (Xtst.shape[0] == X.shape[0]) & (Xtst.shape[1] == X.shape[1])

    X_rated = np.sum(X != 0, axis=1)  # number of total rated items per user
    Xtr_rated = np.sum(Xtr != 0, axis=1)  # number of rated items in the train set
    Xtst_rated = np.sum(Xtst != 0, axis=1)  # number of rated items in the test set

    # global split: check that the all dataset is split in the correct ratio
    assert Xtr_rated.sum() / (X_rated.sum()) == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )

    assert Xtst_rated.sum() / (X_rated.sum()) == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # This implementation of the stratified splitter performs a random split at the single user level. Here we check
    # that also this more stringent condition is verified. Note that user to user fluctuations in the split ratio
    # are stronger than for the entire dataset due to the random nature of the per user splitting.
    # For this reason we allow a slightly bigger tolerance, as specified in the test_specs()

    assert (
        Xtr_rated / X_rated <= test_specs["ratio"] + test_specs["fluctuation"]
    ).all() & (
        Xtr_rated / X_rated >= test_specs["ratio"] - test_specs["fluctuation"]
    ).all()

    assert (
        Xtst_rated / X_rated <= (1 - test_specs["ratio"]) + test_specs["fluctuation"]
    ).all() & (
        Xtst_rated / X_rated >= (1 - test_specs["ratio"]) - test_specs["fluctuation"]
    ).all()


def test_float_numpy_stratified_splitter(test_specs, python_float_dataset):
    # generate a syntetic dataset
    X = python_float_dataset

    # the splitter returns (in order): train and test user/affinity matrices, train and test datafarmes and user/items to matrix maps
    Xtr, Xtst = numpy_stratified_split(
        X, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    # Tests
    # check that the generated matrices have the correct dimensions
    assert (Xtr.shape[0] == X.shape[0]) & (Xtr.shape[1] == X.shape[1])

    assert (Xtst.shape[0] == X.shape[0]) & (Xtst.shape[1] == X.shape[1])

    X_rated = np.sum(X != 0, axis=1)  # number of total rated items per user
    Xtr_rated = np.sum(Xtr != 0, axis=1)  # number of rated items in the train set
    Xtst_rated = np.sum(Xtst != 0, axis=1)  # number of rated items in the test set

    # global split: check that the all dataset is split in the correct ratio
    assert Xtr_rated.sum() / (X_rated.sum()) == pytest.approx(
        test_specs["ratio"], test_specs["tolerance"]
    )

    assert Xtst_rated.sum() / (X_rated.sum()) == pytest.approx(
        1 - test_specs["ratio"], test_specs["tolerance"]
    )

    # This implementation of the stratified splitter performs a random split at the single user level. Here we check
    # that also this more stringent condition is verified. Note that user to user fluctuations in the split ratio
    # are stronger than for the entire dataset due to the random nature of the per user splitting.
    # For this reason we allow a slightly bigger tolerance, as specified in the test_specs()

    assert Xtr_rated / X_rated == pytest.approx(
        test_specs["ratio"], rel=test_specs["fluctuation"]
    )

    assert Xtst_rated / X_rated == pytest.approx(
        (1 - test_specs["ratio"]), rel=test_specs["fluctuation"]
    )
