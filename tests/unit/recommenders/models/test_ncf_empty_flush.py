# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import pytest

from recommenders.models.ncf.dataset import Dataset


@pytest.mark.parametrize("shuffle_size", [None, 2, 4])
@pytest.mark.parametrize("yield_id", [False, True])
def test_train_loader_finishes_after_exact_batches(tmp_path, shuffle_size, yield_id):
    path = tmp_path / "train.csv"
    saved_path = tmp_path / "saved.csv"
    training = pd.DataFrame(
        {"userID": [7, 7, 9, 9], "itemID": [11, 13, 11, 13], "rating": [1.0] * 4}
    )
    training.to_csv(path, index=False)
    dataset = Dataset(path, n_neg=0)

    batches = list(
        dataset.train_loader(
            batch_size=2,
            shuffle_size=shuffle_size,
            yield_id=yield_id,
            write_to=saved_path,
        )
    )

    assert len(batches) == 2
    records = []
    for users, items, ratings in batches:
        if yield_id:
            users = [dataset.id2user[user] for user in users]
            items = [dataset.id2item[item] for item in items]
        records.extend(zip(users, items, ratings))
    assert sorted(records) == sorted(training.itertuples(index=False, name=None))
    saved = pd.read_csv(saved_path)
    assert saved.shape[0] == 4
    assert sorted(saved.itertuples(index=False, name=None)) == sorted(records)


def test_train_loader_still_drops_incomplete_final_batch(tmp_path):
    path = tmp_path / "train.csv"
    pd.DataFrame(
        {"userID": [7, 7, 9, 9, 9], "itemID": [11, 13, 11, 13, 17], "rating": [1.0] * 5}
    ).to_csv(path, index=False)

    batches = list(Dataset(path, n_neg=0).train_loader(2, shuffle_size=2))

    assert len(batches) == 2
    assert all(len(batch[0]) == 2 for batch in batches)
