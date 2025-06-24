import numpy as np
import pytest
from recommenders.models.embdotbias.utils import cartesian_product


def test_cartesian_product_two_arrays():
    a = np.array([1, 2])
    b = np.array([3, 4])
    result = cartesian_product(a, b)
    expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]])
    np.testing.assert_array_equal(result, expected)


def test_cartesian_product_three_arrays():
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    result = cartesian_product(a, b, c)
    expected = np.array([
        [1, 3, 5], [1, 3, 6], [1, 4, 5], [1, 4, 6],
        [2, 3, 5], [2, 3, 6], [2, 4, 5], [2, 4, 6]
    ])
    np.testing.assert_array_equal(result, expected)


def test_cartesian_product_single_array():
    a = np.array([1, 2, 3])
    result = cartesian_product(a)
    expected = np.array([[1], [2], [3]])
    np.testing.assert_array_equal(result, expected)


def test_cartesian_product_empty_array():
    a = np.array([])
    b = np.array([1, 2])
    result = cartesian_product(a, b)
    expected = np.empty((0, 2))
    np.testing.assert_array_equal(result, expected) 