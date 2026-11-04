# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import pytest

from recommenders.models.ncf.dataset import DataFile, Dataset


@pytest.mark.parametrize("test_batch", [None, "test_batch"])
def test_datafile_indexes_one_row(tmp_path, test_batch):
    path = tmp_path / "single.csv"
    row = {"userID": [7], "itemID": [11], "rating": [3.0]}
    if test_batch is not None:
        row[test_batch] = [0]
    pd.DataFrame(row).to_csv(path, index=False)

    datafile = DataFile(path, "userID", "itemID", "rating", test_batch)

    assert datafile.data_len == 1
    assert datafile.user2id == {7: 0}
    assert datafile.item2id == {11: 0}
    assert list(datafile.batch_indices_range) == ([] if test_batch is None else [0])
    with datafile:
        loaded = datafile.load_data(
            7 if test_batch is None else 0, by_user=test_batch is None
        )
    pd.testing.assert_frame_equal(
        loaded, pd.DataFrame({"userID": [7], "itemID": [11], "rating": [1.0]})
    )


def test_dataset_indexes_one_training_interaction(tmp_path):
    path = tmp_path / "single.csv"
    pd.DataFrame({"userID": [7], "itemID": [11], "rating": [3.0]}).to_csv(
        path, index=False
    )

    dataset = Dataset(path, n_neg=0)

    assert dataset.n_users == dataset.n_items == dataset.train_len == 1
    assert dataset.id2user == {0: 7}
    assert dataset.id2item == {0: 11}
