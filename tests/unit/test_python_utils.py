# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import pytest

from reco_utils.common.python_utils import (
    exponential_decay,
    jaccard,
    lift,
    get_top_k_scored_items,
)

TOL = 0.0001


@pytest.fixture
def target_matrices(scope="module"):
    J1 = np.array([[1.0, 0.0, 0.5],
                   [0.0, 1.0, 0.33333],
                   [0.5, 0.33333, 1.0]])
    J2 = np.array([[1.0, 0.0, 0.0, 0.2],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.5],
                   [0.2, 0.0, 0.5, 1.0]])
    L1 = np.array([[1.0, 0.0, 0.5],
                   [0.0, 0.5, 0.25],
                   [0.5, 0.25, 0.5]])
    L2 = np.array([[0.5, 0.0, 0.0, 0.125],
                   [0.0, 0.33333, 0.0, 0.0],
                   [0.0, 0.0, 0.5, 0.25],
                   [0.125, 0.0, 0.25, 0.25]])
    return {
        "jaccard1": pytest.approx(J1, TOL),
        "jaccard2": pytest.approx(J2, TOL),
        "lift1": pytest.approx(L1, TOL),
        "lift2": pytest.approx(L2, TOL)
    }


@pytest.fixture(scope="module")
def python_data():
    cooccurrence1 = np.array([[1.0, 0.0, 1.0],
                              [0.0, 2.0, 1.0],
                              [1.0, 1.0, 2.0]])
    cooccurrence2 = np.array([[2.0, 0.0, 0.0, 1.0],
                              [0.0, 3.0, 0.0, 0.0],
                              [0.0, 0.0, 2.0, 2.0],
                              [1.0, 0.0, 2.0, 4.0]])
    return cooccurrence1, cooccurrence2


def test_python_jaccard(python_data, target_matrices):
    cooccurrence1, cooccurrence2 = python_data
    J1 = jaccard(cooccurrence1)
    assert type(J1) == np.ndarray
    assert J1 == target_matrices["jaccard1"]

    J2 = jaccard(cooccurrence2)
    assert type(J2) == np.ndarray
    assert J2 == target_matrices["jaccard2"]


def test_python_lift(python_data, target_matrices):
    cooccurrence1, cooccurrence2 = python_data
    L1 = lift(cooccurrence1)
    assert type(L1) == np.ndarray
    assert L1 == target_matrices["lift1"]

    L2 = lift(cooccurrence2)
    assert type(L2) == np.ndarray
    assert L2 == target_matrices["lift2"]


def test_exponential_decay():
    values = np.array([1, 2, 3, 4, 5, 6])
    expected = np.array([0.25, 0.35355339, 0.5, 0.70710678, 1., 1.])
    actual = exponential_decay(value=values, max_val=5, half_life=2)
    assert np.allclose(actual, expected, atol=TOL)


def test_get_top_k_scored_items():
    scores = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [1, 5, 3, 4, 2]])
    top_items, top_scores = get_top_k_scored_items(scores=scores, top_k=3, sort_top_k=True)

    assert np.array_equal(top_items, np.array([[4, 3, 2], [0, 1, 2], [1, 3, 2]]))
    assert np.array_equal(top_scores, np.array([[5, 4, 3], [5, 4, 3], [5, 4, 3]]))
