# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest
import numpy as np
from recommenders.utils import python_utils as target
from scipy import sparse


@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("sort", [False, True])
def test_zero_k(sparse_input, sort):
    scores = np.array([[0.2, 0.7, 0.4], [0.6, 0.5, 0.3]])
    if sparse_input:
        scores = sparse.csr_matrix(scores)
    indices, values = target.get_top_k_scored_items(scores, 0, sort)
    assert indices.shape == values.shape == (2, 0)
    assert indices.dtype.kind in "iu"
    assert values.dtype == scores.dtype


@pytest.mark.parametrize("k", [1, 2, 3, 7])
def test_positive_k(k):
    scores = np.array([[0.2, 0.7, 0.4], [0.6, 0.5, 0.3]])
    indices, values = target.get_top_k_scored_items(scores, k, True)
    n = min(k, 3)
    expected = np.argsort(-scores, axis=1)[:, :n]
    np.testing.assert_array_equal(indices, expected)
    np.testing.assert_allclose(values, np.take_along_axis(scores, expected, axis=1))


@pytest.mark.parametrize("shape", [(2, 0), (0, 3), (0, 0)])
def test_empty_axes(shape):
    scores = np.empty(shape, dtype=np.float32)
    i, v = target.get_top_k_scored_items(scores, 0, True)
    assert i.shape == v.shape == (shape[0], 0)
