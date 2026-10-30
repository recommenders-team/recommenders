# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recommenders.utils.python_utils import (
    cosine_similarity,
    inclusion_index,
    jaccard,
    lift,
)


@pytest.mark.parametrize(
    "dtype, count", [(np.uint8, 200), (np.int32, 50000), (np.int64, 4000000000)]
)
@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("metric", [cosine_similarity, inclusion_index, jaccard, lift])
def test_similarity_normalization_does_not_overflow_integer_counts(
    dtype, count, as_sparse, metric
):
    values = np.array([[count, count // 2], [count // 2, count]], dtype=dtype)
    data = csr_matrix(values) if as_sparse else values.copy()
    if metric is jaccard:
        expected = [[1, 1 / 3], [1 / 3, 1]]
    elif metric is lift:
        expected = np.array([[1, 0.5], [0.5, 1]]) / count
    else:
        expected = [[1, 0.5], [0.5, 1]]

    result = metric(data)

    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(data.toarray() if as_sparse else data, values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("metric", [cosine_similarity, inclusion_index, jaccard, lift])
def test_floating_count_precision_is_unchanged(dtype, metric):
    values = np.array([[4, 2], [2, 4]], dtype=dtype)
    result = metric(values)
    assert result.dtype == values.dtype
