# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import collections
import pytest
import numpy as np
from scipy.sparse import csr_matrix

try:
    from recommenders.models.geoimc.geoimc_data import DataPtr
    from recommenders.models.geoimc.geoimc_predict import Inferer
    from recommenders.models.geoimc.geoimc_algorithm import IMCProblem
    from recommenders.models.geoimc.geoimc_utils import (
        length_normalize,
        mean_center,
        reduce_dims,
    )
    from pymanopt.manifolds import Stiefel, SymmetricPositiveDefinite
except:
    pass  # skip if pymanopt not installed


_IMC_TEST_DATA = [
    (
        csr_matrix(np.array([[1, 5, 3], [7, 2, 1]])),
        [
            np.array([[0, 6, 0, 5], [7, 1, 2, 1]]),
            np.array([[8, 8, 0, 8, 4], [7, 4, 3, 0, 7], [0, 6, 8, 7, 2]]),
        ],
    ),
    (
        csr_matrix(np.array([[8, 6, 0], [6, 6, 2]])),
        [
            np.array([[3, 7, 5, 7, 6], [6, 5, 6, 8, 1]]),
            np.array([[5, 3, 2, 7, 6], [2, 6, 9, 3, 3], [9, 9, 4, 9, 2]]),
        ],
    ),
]


# `geoimc_data` tests
@pytest.mark.experimental
@pytest.mark.parametrize("data, entities", _IMC_TEST_DATA)
def test_dataptr(data, entities):
    ptr = DataPtr(data, entities)
    assert (ptr.get_data() != data).nnz == 0
    assert np.array_equal(ptr.get_entity("row"), entities[0])
    assert np.array_equal(ptr.get_entity("col"), entities[1])


# `geoimc_utils` tests
@pytest.mark.experimental
@pytest.mark.parametrize(
    "matrix",
    [
        (np.array([[3, 5, 6], [2, 7, 0], [0, 5, 2]])),
        (np.array([[7, 9, 9], [4, 3, 8], [6, 0, 3]])),
    ],
)
def test_length_normalize(matrix):
    assert np.allclose(
        np.sqrt(np.sum(length_normalize(matrix) ** 2, axis=1)),
        np.ones(matrix.shape[0]),
        atol=1e-6,
    )


@pytest.mark.experimental
@pytest.mark.parametrize(
    "matrix",
    [
        (np.array([[3, 5, 6], [2, 7, 0], [0, 5, 2]], dtype="float64")),
        (np.array([[7, 9, 9], [4, 3, 8], [6, 0, 3]], dtype="float64")),
    ],
)
def test_mean_center(matrix):
    mean_center(matrix)
    assert np.allclose(
        np.mean(matrix, axis=0), np.zeros(matrix.shape[1], dtype="float64"), atol=1e-10
    )


@pytest.mark.experimental
def test_reduce_dims():
    matrix = np.random.rand(100, 100)
    assert reduce_dims(matrix, 50).shape[1] == 50


# `geoimc_algorithm` tests
@pytest.mark.experimental
@pytest.mark.parametrize(
    "dataPtr, rank",
    [
        (DataPtr(_IMC_TEST_DATA[0][0], _IMC_TEST_DATA[0][1]), 3),
        (DataPtr(_IMC_TEST_DATA[1][0], _IMC_TEST_DATA[1][1]), 3),
    ],
)
@pytest.mark.experimental
def test_imcproblem(dataPtr, rank):

    # Test init
    prblm = IMCProblem(dataPtr, rank=rank)
    assert np.array_equal(prblm.X, dataPtr.get_entity("row"))
    assert np.array_equal(prblm.Z, dataPtr.get_entity("col"))
    assert (prblm.Y != dataPtr.get_data()).nnz == 0
    assert prblm.rank == rank
    assert prblm.lambda1 == 1e-2
    assert prblm.W is None
    assert not prblm.optima_reached

    # Test solve
    prblm.solve(10, 10, 0)
    assert len(prblm.W) == 3
    assert prblm.optima_reached

    # Test reset
    prblm.reset()
    assert prblm.W is None
    assert not prblm.optima_reached


# `geoimc_predict` tests
def _inference_factors(dtype):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(4, 8)).astype(dtype)
    z = rng.normal(size=(6, 9)).astype(dtype)
    u = np.linalg.qr(rng.normal(size=(8, 3)))[0].astype(dtype)
    v = np.linalg.qr(rng.normal(size=(9, 3)))[0].astype(dtype)
    q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    b = ((q * [0.1, 1.0, 4.0]) @ q.T).astype(dtype)
    return DataPtr(csr_matrix((4, 6)), [x, z]), [u, b, v]


@pytest.mark.experimental
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("transformation", ["", "mean", "topk"])
@pytest.mark.parametrize("filtered", [False, True])
def test_inferer_factorized_reference(dtype, transformation, filtered):
    data, weights = _inference_factors(dtype)
    if filtered:
        data.entity_indices = [np.array([3, 1]), np.array([4, 0, 2])]
    saved_weights = [weight.copy() for weight in weights]
    x, z = data.get_entity("row"), data.get_entity("col")
    # The trained middle factor is symmetric positive definite. The product of
    # its two square roots recovers it, giving an independent score reference.
    expected = (x @ weights[0]) @ weights[1] @ (z @ weights[2]).T
    if transformation == "mean":
        expected = (expected > expected.mean()).astype(expected.dtype)
    elif transformation == "topk":
        indices = np.argsort(expected, axis=1)[:, -2:]
        expected = np.zeros_like(expected)
        np.put_along_axis(expected, indices, 1, axis=1)
    actual = Inferer(k=2, transformation=transformation).infer(data, weights)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
    for weight, saved in zip(weights, saved_weights):
        np.testing.assert_array_equal(weight, saved)


@pytest.mark.experimental
def test_inferer_uses_updated_middle_factor():
    data, weights = _inference_factors(np.float64)
    inferer = Inferer()
    original = inferer.infer(data, weights)
    weights[1] *= 4
    np.testing.assert_allclose(inferer.infer(data, weights), 4 * original)


@pytest.mark.experimental
def test_inferer_accepts_direct_embeddings():
    rows = np.array([[1.0, 2.0], [3.0, -1.0]])
    columns = np.array([[0.0, 1.0], [2.0, 3.0], [-1.0, 0.5]])
    np.testing.assert_array_equal(
        Inferer().infer([rows, columns], None), rows @ columns.T
    )


@pytest.mark.experimental
def test_inferer_init():
    assert Inferer(method="dot").method.__name__ == "PlainScalarProduct"


@pytest.mark.experimental
@pytest.mark.parametrize(
    "dataPtr",
    [
        DataPtr(_IMC_TEST_DATA[0][0], _IMC_TEST_DATA[0][1]),
        DataPtr(_IMC_TEST_DATA[1][0], _IMC_TEST_DATA[1][1]),
    ],
)
def test_inferer_infer(dataPtr):
    test_data = dataPtr

    rowFeatureDim = test_data.get_entity("row").shape[1]
    colFeatureDim = test_data.get_entity("col").shape[1]
    rank = 2
    W = [
        Stiefel(rowFeatureDim, rank).rand(),
        SymmetricPositiveDefinite(rank).rand(),
        Stiefel(colFeatureDim, rank).rand(),
    ]

    Inferer(method="dot").infer(test_data, W)

    inference = Inferer(method="dot", transformation="mean").infer(test_data, W)
    nOccurences = collections.Counter(inference.ravel())
    assert nOccurences[0] + nOccurences[1] == inference.size

    k = 2
    inference = Inferer(method="dot", k=k, transformation="topk").infer(test_data, W)
    nOccurences = collections.Counter(inference.ravel())
    assert nOccurences[0] + nOccurences[1] == inference.size
    assert np.max(np.count_nonzero(inference == 1, axis=0)) <= k
