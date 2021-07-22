# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    SEED,
)
from recommenders.models.ncf.dataset import Dataset


N_NEG = 5
N_NEG_TEST = 10
BATCH_SIZE = 32


def test_data_preprocessing(python_dataset_ncf):
    train, test = python_dataset_ncf
    data = Dataset(
        train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST, seed=SEED
    )

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


def test_train_loader(python_dataset_ncf):
    train, test = python_dataset_ncf
    data = Dataset(
        train=train, test=test, n_neg=N_NEG, n_neg_test=N_NEG_TEST, seed=SEED
    )

    # collect positvie user-item dict
    positive_pool = {}
    for u in train[DEFAULT_USER_COL].unique():
        positive_pool[u] = set(train[train[DEFAULT_USER_COL] == u][DEFAULT_ITEM_COL])

    # without negative sampling
    for batch in data.train_loader(batch_size=BATCH_SIZE, shuffle=False):
        user, item, labels = batch
        # shape
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


def test_test_loader(python_dataset_ncf):
    train, test = python_dataset_ncf
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
