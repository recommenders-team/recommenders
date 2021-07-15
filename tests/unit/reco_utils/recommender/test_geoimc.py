# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import collections
import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from pandas.testing import assert_frame_equal

from reco_utils.utils.python_utils import binarize
from reco_utils.recommender.geoimc.geoimc_data import DataPtr
from reco_utils.recommender.geoimc.geoimc_predict import PlainScalarProduct, Inferer
from reco_utils.recommender.geoimc.geoimc_algorithm import IMCProblem
from reco_utils.recommender.geoimc.geoimc_utils import (
    length_normalize,
    mean_center,
    reduce_dims,
)
from pymanopt.manifolds import Stiefel, PositiveDefinite

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
@pytest.mark.parametrize("data, entities", _IMC_TEST_DATA)
def test_dataptr(data, entities):
    ptr = DataPtr(data, entities)
    assert (ptr.get_data() != data).nnz == 0
    assert np.array_equal(ptr.get_entity("row"), entities[0])
    assert np.array_equal(ptr.get_entity("col"), entities[1])


# `geoimc_utils` tests
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


def test_reduce_dims():
    matrix = np.random.rand(100, 100)
    assert reduce_dims(matrix, 50).shape[1] == 50


# `geoimc_algorithm` tests
@pytest.mark.parametrize(
    "dataPtr, rank",
    [
        (DataPtr(_IMC_TEST_DATA[0][0], _IMC_TEST_DATA[0][1]), 3),
        (DataPtr(_IMC_TEST_DATA[1][0], _IMC_TEST_DATA[1][1]), 3),
    ],
)
def test_imcproblem(dataPtr, rank):

    # Test init
    prblm = IMCProblem(dataPtr, rank=rank)
    assert np.array_equal(prblm.X, dataPtr.get_entity("row"))
    assert np.array_equal(prblm.Z, dataPtr.get_entity("col"))
    assert (prblm.Y != dataPtr.get_data()).nnz == 0
    assert prblm.rank == rank
    assert prblm.lambda1 == 1e-2
    assert prblm.W is None
    assert prblm.optima_reached == False

    # Test solve
    prblm.solve(10, 10, 0)
    assert len(prblm.W) == 3
    assert prblm.optima_reached

    # Test reset
    prblm.reset()
    assert prblm.W is None
    assert prblm.optima_reached == False


# `geoimc_predict` tests
def test_inferer_init():
    assert Inferer(method="dot").method.__name__ == "PlainScalarProduct"


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
        PositiveDefinite(rank).rand(),
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
