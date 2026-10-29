# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from recommenders.datasets.python_splitters import numpy_stratified_split


@pytest.mark.parametrize(
    "ratio, expected_counts",
    [
        (0.8, [40, 80]),
        (0.995, [1, 2]),
        (0.333, [133, 267]),
        (0.55, [90, 180]),
        (0.75, [50, 100]),
    ],
)
@pytest.mark.parametrize("dtype", [np.int64, np.float64])
def test_numpy_split_uses_full_ratio_precision(ratio, expected_counts, dtype):
    values = np.zeros((2, 400), dtype=dtype)
    values[0, :200] = np.arange(1, 201)
    values[1] = np.arange(1, 401)
    original = values.copy()

    train, test = numpy_stratified_split(values, ratio=ratio, seed=31)

    np.testing.assert_array_equal(np.count_nonzero(test, axis=1), expected_counts)
    np.testing.assert_array_equal(train + test, original)
    np.testing.assert_array_equal(values, original)
    assert not np.any((train != 0) & (test != 0))
    assert train.dtype == test.dtype == values.dtype
    repeated = numpy_stratified_split(values, ratio=ratio, seed=31)
    np.testing.assert_array_equal(train, repeated[0])
    np.testing.assert_array_equal(test, repeated[1])
