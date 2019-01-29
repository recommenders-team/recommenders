# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Test common python utils
"""
import numpy as np
import pytest

from scipy.sparse import csc, csc_matrix
from reco_utils.common.python_utils import (
    jaccard,
    lift
)

TOL = 0.0001


@pytest.fixture
def target_matrices():
    J1 = np.mat('1.0, 0.0, 0.5; '
                '0.0, 1.0, 0.33333; '
                '0.5, 0.33333, 1.0')
    J2 = np.mat('1.0, 0.0, 0.0, 0.2; '
                '0.0, 1.0, 0.0, 0.0; '
                '0.0, 0.0, 1.0, 0.5; '
                '0.2, 0.0, 0.5, 1.0')
    L1 = np.mat('1.0, 0.0, 0.5; '
                '0.0, 0.5, 0.25; '
                '0.5, 0.25, 0.5')
    L2 = np.mat('0.5, 0.0, 0.0, 0.125; '
                '0.0, 0.33333, 0.0, 0.0; '
                '0.0, 0.0, 0.5, 0.25; '
                '0.125, 0.0, 0.25, 0.25')
    return {
        "jaccard1": pytest.approx(J1, TOL),
        "jaccard2": pytest.approx(J2, TOL),
        "lift1": pytest.approx(L1, TOL),
        "lift2": pytest.approx(L2, TOL)
    }


@pytest.fixture(scope="module")
def python_data():
    D1 = np.mat('1.0, 0.0, 1.0; '
                '0.0, 2.0, 1.0; '
                '1.0, 1.0, 2.0')
    cooccurrence1 = csc_matrix(D1)
    D2 = np.mat('2.0, 0.0, 0.0, 1.0; '
                '0.0, 3.0, 0.0, 0.0; '
                '0.0, 0.0, 2.0, 2.0; '
                '1.0, 0.0, 2.0, 4.0')
    cooccurrence2 = csc_matrix(D2)

    return cooccurrence1, cooccurrence2


def test_python_jaccard(python_data, target_matrices):
    cooccurrence1, cooccurrence2 = python_data
    J1 = jaccard(cooccurrence1)
    assert type(J1) == csc.csc_matrix
    assert J1.todense() == target_matrices["jaccard1"]

    J2 = jaccard(cooccurrence2)
    assert type(J2) == csc.csc_matrix
    assert J2.todense() == target_matrices["jaccard2"]


def test_python_lift(python_data, target_matrices):
    cooccurrence1, cooccurrence2 = python_data
    L1 = lift(cooccurrence1)
    assert type(L1) == csc.csc_matrix
    assert L1.todense() == target_matrices["lift1"]

    L2 = lift(cooccurrence2)
    assert type(L2) == csc.csc_matrix
    assert L2.todense() == target_matrices["lift2"]
