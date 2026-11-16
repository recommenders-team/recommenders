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


@pytest.fixture
def geoimc_sparse_case():
    """A sparse observed matrix with distinct row and column feature dimensions."""
    rng = np.random.default_rng(812)
    row = rng.normal(size=(8, 4))
    col = rng.normal(size=(7, 5))
    row[np.abs(row) < 0.7] = 0
    col[np.abs(col) < 0.7] = 0
    ratings = rng.normal(size=(8, 7))
    ratings[rng.random((8, 7)) < 0.4] = 0
    target = csr_matrix(ratings)
    weights = [
        np.linalg.qr(rng.normal(size=(4, 2)))[0],
        np.array([[1.3, 0.2], [0.2, 1.7]]),
        np.linalg.qr(rng.normal(size=(5, 2)))[0],
    ]
    return target, row, col, weights


def _geoimc_dense_objective(target, row, col, weights, regularizer):
    """Independent full-matrix objective at only the observed entries."""
    observed = target.tocoo()
    predictions = row @ weights[0] @ weights[1] @ weights[2].T @ col.T
    residuals = predictions[observed.row, observed.col] - observed.data
    return 0.5 * np.mean(residuals**2) + 0.5 * regularizer * np.sum(weights[1] ** 2)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "row_sparse,col_sparse",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_geoimc_sparse_feature_objective_and_gradients(
    geoimc_sparse_case, row_sparse, col_sparse
):
    target, row, col, weights = geoimc_sparse_case
    features = [
        csr_matrix(row) if row_sparse else row,
        csr_matrix(col) if col_sparse else col,
    ]
    problem = IMCProblem(DataPtr(target, features), rank=2, lambda1=0.03)
    residual = np.zeros(target.nnz)
    cost = problem._cost(weights, residual)
    expected = _geoimc_dense_objective(target, row, col, weights, problem.lambda1)
    np.testing.assert_allclose(cost, expected, rtol=1e-12, atol=1e-12)
    gradients = problem._egrad(weights, residual)
    for factor, (parameter, actual) in enumerate(zip(weights, gradients)):
        assert actual.dtype == np.float64
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            perturbation = np.zeros_like(parameter)
            perturbation[index] = 1e-5
            if factor == 1:
                # The middle factor is symmetric positive definite.
                perturbation = (perturbation + perturbation.T) / 2
            plus = [value.copy() for value in weights]
            minus = [value.copy() for value in weights]
            plus[factor] += perturbation
            minus[factor] -= perturbation
            numerical[index] = (
                _geoimc_dense_objective(target, row, col, plus, problem.lambda1)
                - _geoimc_dense_objective(target, row, col, minus, problem.lambda1)
            ) / 2e-5
        np.testing.assert_allclose(actual, numerical, rtol=2e-7, atol=2e-9)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "row_sparse,col_sparse",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_geoimc_sparse_features_fit_and_cold_start(
    geoimc_sparse_case, row_sparse, col_sparse
):
    target, row, col, weights = geoimc_sparse_case
    features = [
        csr_matrix(row) if row_sparse else row.copy(),
        csr_matrix(col) if col_sparse else col.copy(),
    ]
    train = DataPtr(target, features)
    train.data_indices = np.array([0, 2, 3, 5, 6])
    train.entity_indices[0] = train.data_indices
    dense_train = DataPtr(target[train.data_indices], [row[train.data_indices], col])
    models = [IMCProblem(data, rank=2, lambda1=0.03) for data in [train, dense_train]]
    initial_cost = _geoimc_dense_objective(
        dense_train.get_data(), dense_train.get_entity("row"), col, weights, 0.03
    )
    for model in models:
        model.W = [value.copy() for value in weights]
        model.solve(30, 8, 0)
        assert model.optima_reached
        assert all(np.isfinite(value).all() for value in model.W)
    for actual, expected in zip(models[0].W, models[1].W):
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-10)
    final_cost = _geoimc_dense_objective(
        dense_train.get_data(), dense_train.get_entity("row"), col, models[0].W, 0.03
    )
    assert final_cost < initial_cost
    held_out = np.array([1, 4, 7])
    test = DataPtr(target, features)
    test.data_indices = held_out
    test.entity_indices[0] = held_out
    scores = Inferer().infer(test, models[0].W)
    expected = (
        row[held_out] @ models[1].W[0] @ models[1].W[1] @ models[1].W[2].T @ col.T
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-8, atol=1e-10)
    for actual, expected in zip(features, [row, col]):
        np.testing.assert_array_equal(
            actual.toarray() if hasattr(actual, "toarray") else actual, expected
        )
    for sparse, actual in zip([row_sparse, col_sparse], features):
        if sparse:
            assert isinstance(actual, csr_matrix)
