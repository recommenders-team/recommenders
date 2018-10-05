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


@pytest.mark.integration
def test_basic_structures(basic_structures):
    assert basic_structures["int"] == 5
    assert basic_structures["yes"] is True
    assert basic_structures["no"] is False
    assert basic_structures["float"] == 0.5
    assert basic_structures["string"] == "Miguel"
    assert basic_structures["none"] is None


@pytest.mark.integration
def test_comparing_numbers(basic_structures):
    assert basic_structures["pi"] == pytest.approx(3.1415926, 0.0000001)
    assert basic_structures["pi"] != pytest.approx(3.1415926, 0.00000001)
    assert basic_structures["int"] > 3
    assert basic_structures["int"] >= 5
    assert basic_structures["int"] < 10
    assert basic_structures["int"] <= 5
