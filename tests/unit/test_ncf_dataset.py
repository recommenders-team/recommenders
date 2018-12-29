
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

from reco_utils.recommender.ncf.dataset import Dataset

N_NEG = 5
N_NEG_TEST = 10
BATCH_SIZE = 32

# data generation

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

    np.random.seed(test_specs["seed"])

    rating = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.randint(
                1, 100, test_specs["number_of_rows"]
            ),
            DEFAULT_ITEM_COL: np.random.randint(
                1, 100, test_specs["number_of_rows"]
            ),
            DEFAULT_RATING_COL: np.random.randint(
                1, 5, test_specs["number_of_rows"]
            ),
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01", test_specs["number_of_rows"]
            ),
        }
    )

    train, test = python_chrono_split(rating, ratio=np.random.choice(test_specs["ratios"]))

    return (train, test)


def test_data_preprocessing(python_dataset):
    # test dataset._data_preprocessing and dataset._reindex

    train, test = python_dataset
    
    data = Dataset(train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST)
    
    # shape
    assert len(data.train) == len(train)
    assert len(data.test) == len(test)

    # index correctness for id2user, user2id, id2item, item2id
    for data_row, row in zip(data.train.iterrows(), train.iterrows()):
        assert data_row[1][DEFAULT_USER_COL] == data.user2id[row[1][DEFAULT_USER_COL]]
        assert row[1][DEFAULT_USER_COL] == data.id2user[data_row[1][DEFAULT_USER_COL]]
        assert data_row[1][DEFAULT_ITEM_COL] == data.item2id[row[1][DEFAULT_ITEM_COL]]
        assert row[1][DEFAULT_ITEM_COL] == data.id2item[data_row[1][DEFAULT_ITEM_COL]]

    for data_row, row in zip(data.test.iterrows(), test.iterrows()):
        assert data_row[1][DEFAULT_USER_COL] == data.user2id[row[1][DEFAULT_USER_COL]]
        assert row[1][DEFAULT_USER_COL] == data.id2user[data_row[1][DEFAULT_USER_COL]]
        assert data_row[1][DEFAULT_ITEM_COL] == data.item2id[row[1][DEFAULT_ITEM_COL]]
        assert row[1][DEFAULT_ITEM_COL] == data.id2item[data_row[1][DEFAULT_ITEM_COL]]

def test_train_loader(python_dataset):
    # test dataset.train_loader()

    train, test = python_dataset
    
    data = Dataset(train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST)

    # collect positvie user-item dict
    positive_pool = {}
    for u in train[DEFAULT_USER_COL].unique():
        positive_pool[u] = set(train[train[DEFAULT_USER_COL] == u][DEFAULT_ITEM_COL])

    # without negative sampling
    for batch in data.train_loader(batch_size=BATCH_SIZE, shuffle=False):
        user, item, labels = batch
        #shape
        assert len(user) == BATCH_SIZE
        assert len(item) == BATCH_SIZE
        assert len(labels) == BATCH_SIZE

        assert max(labels) == min(labels)

        # right labels
        for u, i, is_pos in zip(user, item, labels):
            if is_pos:
                assert i in positive_pool[u] 
            else: 
                assert i not in positive_pool[u] 

    data.negative_sampling()

    label_list = []

    batches = []


    for idx, batch in enumerate(data.train_loader(batch_size=1)):
        user, item, labels = batch
        assert len(user) == 1
        assert len(item) == 1
        assert len(labels) == 1

        # right labels
        for u, i, is_pos in zip(user, item, labels):
            if is_pos:
                assert i in positive_pool[u] 
            else: 
                assert i not in positive_pool[u] 

            label_list.append(is_pos)

    # neagtive smapling
    assert len(label_list) == (N_NEG + 1) * sum(label_list)


def test_test_loader(python_dataset):
    # test for dataset.test_loader()

    train, test = python_dataset
    
    data = Dataset(train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST)

    # positive user-item dict, noting that the pool is train+test
    positive_pool = {}
    df = train.append(test)
    for u in df[DEFAULT_USER_COL].unique():
        
        positive_pool[u] = set(df[df[DEFAULT_USER_COL] == u][DEFAULT_ITEM_COL])

    for batch in data.test_loader():
        user, item, labels = batch
        # shape

        assert len(user) == N_NEG_TEST + 1
        assert len(item) == N_NEG_TEST + 1
        assert len(labels) == N_NEG_TEST + 1

        label_list = []

        for u, i, is_pos in zip(user, item, labels):
            if is_pos:
                assert i in positive_pool[u] 
            else: 
                assert i not in positive_pool[u] 

            label_list.append(is_pos)

        # leave-one-out
        assert sum(label_list) == 1
        # right labels
        assert len(label_list) == (N_NEG_TEST + 1) * sum(label_list)
