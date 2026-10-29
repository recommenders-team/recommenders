# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.datasets.python_splitters import (
    python_chrono_split,
    python_stratified_split,
)


@pytest.mark.parametrize("random", [False, True])
@pytest.mark.parametrize("min_rating", [1, 2])
@pytest.mark.parametrize("custom_columns", [False, True])
def test_stratified_split_preserves_all_input_columns(
    random, min_rating, custom_columns
):
    data = pd.DataFrame(
        {
            "userID": ["b", "a", "b", "a", "b", "a"],
            "itemID": [0, 1, 2, 3, 4, 5],
            "timestamp": [3, 2, 1, 3, 2, 1],
            "count": [100, 101, 102, 103, 104, 105],
            "rank": [10, 11, 12, 13, 14, 15],
            "random": list("ABCDEF"),
        },
        index=[7, 7, 2, 4, 4, 1],
    )
    col_user, col_item, col_timestamp = "userID", "itemID", "timestamp"
    if custom_columns:
        data = data.drop(columns=["count", "rank", "random"]).rename(
            columns={"userID": "random", "itemID": "count", "timestamp": "rank"}
        )
        col_user, col_item, col_timestamp = "random", "count", "rank"
    original = data.copy(deep=True)
    options = dict(
        ratio=[1 / 3, 2 / 3],
        min_rating=min_rating,
        col_user=col_user,
        col_item=col_item,
    )
    if random:
        splits = python_stratified_split(data, seed=19, **options)
    else:
        splits = python_chrono_split(data, col_timestamp=col_timestamp, **options)

    pd.testing.assert_frame_equal(data, original)
    for split in splits:
        assert list(split.columns) == list(original.columns)
    pd.testing.assert_frame_equal(
        pd.concat(splits).sort_values(col_item),
        original.sort_values(col_item),
    )
    assert [len(split) for split in splits] == [2, 4]
    if not random:
        assert splits[0][col_timestamp].tolist() == [1, 1]


def test_random_split_preserves_seeded_row_order():
    data = pd.DataFrame({"userID": [1, 0, 1, 0, 1, 0], "itemID": np.arange(6)})
    expected = data.assign(draw=np.random.RandomState(23).rand(len(data)))
    expected = expected.sort_values(["userID", "draw"]).groupby("userID").head(2)
    train, _ = python_stratified_split(data, ratio=2 / 3, seed=23)
    assert train.index.tolist() == expected.index.tolist()
