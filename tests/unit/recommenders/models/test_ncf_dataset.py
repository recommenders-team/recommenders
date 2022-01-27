# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import pandas as pd
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
from recommenders.models.ncf.dataset import (
    DataFile,
    NegativeSampler,
    Dataset,
    EmptyFileException,
    MissingFieldsException,
    FileNotSortedException,
    MissingUserException
)


@pytest.mark.gpu
def test_datafile_init(dataset_ncf_files_sorted):
    train_path, _, _ = dataset_ncf_files_sorted
    train = pd.read_csv(train_path)
    users = train[DEFAULT_USER_COL].unique()
    items = train[DEFAULT_ITEM_COL].unique()
    datafile = DataFile(
        train_path, DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, col_test_batch=None, binary=True
    )
    assert set(datafile.users) == set(users)
    assert set(datafile.items) == set(items)
    assert set(datafile.user2id.keys()) == set(users)
    assert set(datafile.item2id.keys()) == set(items)
    assert len(set(datafile.user2id.values())) == len(users)
    assert len(set(datafile.item2id.values())) == len(items)
    assert datafile.data_len == train.shape[0]

    datafile_records = []
    with datafile as f:
        for line in f:
            datafile_records.append({
                DEFAULT_USER_COL: line[DEFAULT_USER_COL],
                DEFAULT_ITEM_COL: line[DEFAULT_ITEM_COL],
                DEFAULT_RATING_COL: line[DEFAULT_RATING_COL]
            })
    datafile_df = pd.DataFrame.from_records(datafile_records)
    assert datafile_df.shape[0] == train.shape[0]

    # test the data loaded from the file is the same as original data
    datafile_df = datafile_df.sort_values(by=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
    train = train.sort_values(by=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])
    train[DEFAULT_RATING_COL] = train[DEFAULT_RATING_COL].apply(lambda x: float(x > 0))
    train = train.drop(DEFAULT_TIMESTAMP_COL, axis=1)
    assert train.equals(datafile_df)

    # test data can be loaded for a valid user and it throws exception for invalid user
    user = train[DEFAULT_USER_COL].iloc[0]
    missing_user = train[DEFAULT_USER_COL].iloc[-1] + 1
    with datafile as f:
        user_data = f.load_data(user)
        assert user_data[DEFAULT_USER_COL].iloc[0] == user
        with pytest.raises(MissingUserException):
            user_data == f.load_data(missing_user)


@pytest.mark.gpu
def test_datafile_init_unsorted(dataset_ncf_files_unsorted):
    train_path, _, _= dataset_ncf_files_unsorted
    with pytest.raises(FileNotSortedException):
        datafile = DataFile(
            train_path, DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, col_test_batch=None, binary=True
        )


@pytest.mark.gpu
def test_datafile_init_empty(dataset_ncf_files_empty):
    train_path, _, _= dataset_ncf_files_empty
    with pytest.raises(EmptyFileException):
        datafile = DataFile(
            train_path, DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, col_test_batch=None, binary=True
        )


@pytest.mark.gpu
def test_datafile_missing_column(dataset_ncf_files_missing_column):
    train_path, _, _= dataset_ncf_files_missing_column
    with pytest.raises(MissingFieldsException):
        datafile = DataFile(
            train_path, DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, col_test_batch=None, binary=True
        )


@pytest.mark.gpu
def test_negative_sampler(caplog):
    user = 1
    n_samples = 3
    user_positive_item_pool = {1, 2}
    item_pool = {1, 2, 3, 4, 5}
    sample_with_replacement = False
    sampler = NegativeSampler(user, n_samples, user_positive_item_pool, item_pool, sample_with_replacement)
    assert sampler.n_samples == 3
    samples = sampler.sample()
    assert set(samples) == item_pool.difference(user_positive_item_pool)

    # test sampler adjusts n_samples down if population is too small and that it raises a warning
    n_samples = 4
    sampler = NegativeSampler(user, n_samples, user_positive_item_pool, item_pool, sample_with_replacement)
    assert sampler.n_samples == 3
    assert "The population of negative items to sample from is too small for user 1" in caplog.text

    # test sampling with replacement returns requested number of samples despite small population
    sampler = NegativeSampler(user, n_samples, user_positive_item_pool, item_pool, sample_with_replacement=True)
    assert sampler.n_samples == 4
    assert len(sampler.sample()) == n_samples


@pytest.mark.gpu
def test_train_loader(tmp_path, dataset_ncf_files_sorted):
    train_path, _, _ = dataset_ncf_files_sorted
    train = pd.read_csv(train_path)
    users = train[DEFAULT_USER_COL].unique()
    items = train[DEFAULT_ITEM_COL].unique()

    n_neg = 1
    dataset = Dataset(train_path, n_neg=n_neg)
    assert dataset.n_users == len(users)
    assert dataset.n_items == len(items)
    assert set(dataset.user2id.keys()) == set(users)
    assert set(dataset.item2id.keys()) == set(items)
    assert len(set(dataset.user2id.values())) == len(users)
    assert len(set(dataset.item2id.values())) == len(items)

    # test number of batches and data size is as expected after loading all training data
    full_data_len = train.shape[0] * 2
    batch_size = full_data_len // 10
    expected_batches = full_data_len // batch_size
    train_save_path = os.path.join(tmp_path, "train_full.csv")
    batch_records = []
    for batch in dataset.train_loader(batch_size, shuffle_size=batch_size, yield_id=True, write_to=train_save_path):
        assert type(batch[0][0]) == int
        assert type(batch[1][0]) == int
        assert type(batch[2][0]) == float
        batch_data = {
            DEFAULT_USER_COL: [dataset.id2user[user] for user in batch[0]],
            DEFAULT_ITEM_COL: [dataset.id2item[item] for item in batch[1]],
            DEFAULT_RATING_COL: batch[2]
        }
        batch_records.append(pd.DataFrame(batch_data))
    
    assert len(batch_records) == expected_batches
    train_loader_df = pd.concat(batch_records).reset_index(drop=True)
    assert train_loader_df.shape[0] == expected_batches * batch_size
    assert set(train_loader_df[DEFAULT_USER_COL]) == set(users)
    assert set(train_loader_df[DEFAULT_ITEM_COL]) == set(items)

    # test that data is successfully saved
    assert os.path.exists(train_save_path)
    train_file_data = pd.read_csv(train_save_path)
    assert train_file_data.equals(train_loader_df)


@pytest.mark.gpu
def test_test_loader(dataset_ncf_files_sorted):
    train_path, _, leave_one_out_test_path = dataset_ncf_files_sorted
    leave_one_out_test = pd.read_csv(leave_one_out_test_path)
    test_users = leave_one_out_test[DEFAULT_USER_COL].unique()

    n_neg = 1
    n_neg_test = 1
    dataset = Dataset(train_path, test_file=leave_one_out_test_path, n_neg=n_neg, n_neg_test=n_neg_test)
    assert set(dataset.test_full_datafile.users) == set(test_users)

    # test number of batches and data size is as expected after loading all test data
    expected_test_batches = leave_one_out_test.shape[0]
    assert max(dataset.test_full_datafile.batch_indices_range) + 1 == expected_test_batches
    batch_records = []
    for batch in dataset.test_loader(yield_id=True):
        assert type(batch[0][0]) == int
        assert type(batch[1][0]) == int
        assert type(batch[2][0]) == float
        batch_data = {
            DEFAULT_USER_COL: [dataset.id2user[user] for user in batch[0]],
            DEFAULT_ITEM_COL: [dataset.id2item[item] for item in batch[1]],
            DEFAULT_RATING_COL: batch[2]
        }
        batch_records.append(pd.DataFrame(batch_data))
    
    assert len(batch_records) == expected_test_batches
    test_loader_df = pd.concat(batch_records).reset_index(drop=True)
    assert test_loader_df.shape[0] == expected_test_batches * n_neg_test * 2
    assert set(test_loader_df[DEFAULT_USER_COL]) == set(test_users)
