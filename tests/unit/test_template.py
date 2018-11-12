# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# More information: https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/
import pytest
import pandas as pd
import numpy as np
from collections import Counter


@pytest.fixture()
def basic_structures():
    data = {
        "int": 5,
        "yes": True,
        "no": False,
        "float": 0.5,
        "pi": 3.141592653589793238462643383279,
        "string": "Miguel",
        "none": None,
    }
    return data


@pytest.fixture()
def complex_structures():
    my_list = [1, 2, 3]
    my_dict = {"a": 1, "b": 2}
    return my_list, my_dict


@pytest.fixture()
def numeric_libs():
    l, d = complex_structures()
    np_array = np.array(l)
    df = pd.DataFrame(d, index=[0])
    series = pd.Series(l)
    return np_array, df, series


def test_basic_structures(basic_structures):
    assert basic_structures["int"] == 5
    assert basic_structures["yes"] is True
    assert basic_structures["no"] is False
    assert basic_structures["float"] == 0.5
    assert basic_structures["string"] == "Miguel"
    assert basic_structures["none"] is None


def test_comparing_numbers(basic_structures):
    assert basic_structures["pi"] == pytest.approx(3.1415926, 0.0000001)
    assert basic_structures["pi"] != pytest.approx(3.1415926, 0.00000001)
    assert basic_structures["int"] > 3
    assert basic_structures["int"] >= 5
    assert basic_structures["int"] < 10
    assert basic_structures["int"] <= 5


def test_lists(complex_structures):
    l = complex_structures[0]
    assert l == [1, 2, 3]
    assert Counter(l) == Counter([2, 1, 3])  # list have same elements
    assert 1 in l
    assert 5 not in l
    assert all(x in l for x in [2, 3])  # sublist in list


def test_dictionaries(complex_structures):
    d = complex_structures[1]
    assert d == {"a": 1, "b": 2}
    assert "a" in d
    assert d.items() <= {"a": 1, "b": 2, "c": 3}.items()  # subdict in dict
    with pytest.raises(KeyError):
        value = d["c"]


def test_pandas(numeric_libs):
    _, df, series = numeric_libs
    df_target = pd.DataFrame({"a": 1, "b": 2}, index=[0])
    series_target = pd.Series([1, 2, 3])
    pd.testing.assert_frame_equal(df, df_target)
    pd.testing.assert_series_equal(series, series_target)


def test_numpy(numeric_libs):
    np_array = numeric_libs[0]
    np_target = np.array([1, 2, 3])
    np_target2 = np.array([0.9999, 2, 3])
    assert np.all(np_array == np_target)
    np.testing.assert_array_equal(np_array, np_target)  # same as before
    np.testing.assert_array_almost_equal(np_array, np_target2, decimal=4)
