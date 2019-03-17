import pytest

from reco_utils.evaluation.parameter_sweep import generate_param_grid


@pytest.fixture(scope="module")
def parameter_dictionary():
    params = {
        "param1": [1, 2, 3],
        "param2": [4, 5, 6],
        "param3": 1
    }

    return params


def test_param_sweep(parameter_dictionary):
    params_grid = generate_param_grid(parameter_dictionary)

    assert params_grid == [
        {'param1': 1, 'param2': 4, 'param3': 1}, {'param1': 1, 'param2': 5, 'param3': 1},
        {'param1': 1, 'param2': 6, 'param3': 1}, {'param1': 2, 'param2': 4, 'param3': 1},
        {'param1': 2, 'param2': 5, 'param3': 1}, {'param1': 2, 'param2': 6, 'param3': 1},
        {'param1': 3, 'param2': 4, 'param3': 1}, {'param1': 3, 'param2': 5, 'param3': 1},
        {'param1': 3, 'param2': 6, 'param3': 1}
    ]
