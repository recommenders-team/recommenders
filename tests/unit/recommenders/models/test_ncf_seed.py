# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import random

import numpy as np
import pandas as pd
import pytest

from recommenders.models.ncf.dataset import Dataset


@pytest.fixture(autouse=True)
def restore_random_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)


@pytest.fixture
def ncf_files(tmp_path):
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    pd.DataFrame(
        {"userID": np.repeat(np.arange(4), 3), "itemID": np.arange(12), "rating": 1.0}
    ).to_csv(train, index=False)
    pd.DataFrame(
        {"userID": np.arange(4), "itemID": [3, 6, 9, 0], "rating": 1.0}
    ).to_csv(test, index=False)
    return train, test


@pytest.mark.parametrize("replacement", [False, True])
def test_seed_reproduces_generated_test_negatives(tmp_path, ncf_files, replacement):
    train, test = ncf_files
    outputs = []
    for index, global_seed in enumerate([11, 99]):
        random.seed(global_seed)
        output = tmp_path / f"test_full_{index}.csv"
        Dataset(
            train,
            test_file=test,
            test_file_full=output,
            n_neg_test=4,
            sample_with_replacement=replacement,
            seed=0,
        )
        outputs.append(output.read_text())

    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("seed", [0, 17])
def test_seed_reproduces_training_shuffle(ncf_files, seed):
    train, _ = ncf_files
    outputs = []
    for global_seed in [11, 99]:
        np.random.seed(global_seed)
        dataset = Dataset(train, n_neg=0, seed=seed)
        outputs.append(list(dataset.train_loader(batch_size=5, shuffle_size=8)))

    assert outputs[0] == outputs[1]
    assert len(outputs[0]) == 2


@pytest.mark.parametrize("replacement", [False, True])
def test_other_datasets_do_not_change_training_samples(ncf_files, replacement):
    train, _ = ncf_files
    first = Dataset(train, n_neg=1, seed=24, sample_with_replacement=replacement)
    other = Dataset(train, n_neg=1, seed=2, sample_with_replacement=replacement)
    reference = Dataset(train, n_neg=1, seed=24, sample_with_replacement=replacement)

    list(other.train_loader(batch_size=5, shuffle_size=10))
    actual = list(first.train_loader(batch_size=5, shuffle_size=10))
    expected = list(reference.train_loader(batch_size=5, shuffle_size=10))

    assert actual == expected
    assert len(actual) == 4


def test_dataset_construction_preserves_global_random_state(ncf_files):
    train, _ = ncf_files
    random.seed(41)
    np.random.seed(43)
    expected_python_state = random.getstate()
    expected_numpy_draw = np.random.RandomState(43).random_sample(5)

    Dataset(train, n_neg=0, seed=7)

    assert random.getstate() == expected_python_state
    np.testing.assert_array_equal(np.random.random_sample(5), expected_numpy_draw)
