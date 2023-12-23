# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest
import nbclient
from pathlib import Path

from recommenders.utils.notebook_utils import (
    is_jupyter,
    is_databricks,
    execute_notebook,
    read_notebook,
    _update_parameters,
)


@pytest.fixture(scope="function")
def notebook_types():
    return Path(__file__).absolute().parent.joinpath("test_notebook_utils.ipynb")


@pytest.fixture(scope="function")
def notebook_programmatic():
    return (
        Path(__file__)
        .absolute()
        .parent.joinpath("programmatic_notebook_execution.ipynb")
    )


@pytest.mark.notebooks
def test_is_jupyter(notebook_types, output_notebook, kernel_name):
    # Test on the terminal
    assert is_jupyter() is False
    assert is_databricks() is False

    # Test on Jupyter notebook
    execute_notebook(
        notebook_types,
        output_notebook,
        kernel_name=kernel_name,
    )
    results = read_notebook(output_notebook)

    assert results["is_jupyter"]
    assert not results["is_databricks"]


@pytest.mark.spark
@pytest.mark.notebooks
@pytest.mark.skip(reason="TODO: Implement this")
def test_is_databricks():
    pass


@pytest.mark.notebooks
def test_update_parameters():
    parameter_cell_source = '''
# Integer
TOP_K = 10
# Float
LEARNING_RATE = 0.001
# String
MOVIELENS_DATA_SIZE = "100k"
# List
RANKING_METRICS = [ evaluator.ndcg_at_k.__name__, evaluator.precision_at_k.__name__ ]
# Boolean
EVALUATE_WHILE_TRAINING = True
'''

    new_parameters = {
        "MOVIELENS_DATA_SIZE": "1m",
        "TOP_K": 1,
        "EVALUATE_WHILE_TRAINING": False,
        "RANKING_METRICS": ["ndcg_at_k", "precision_at_k"],
        "LEARNING_RATE": 0.1,
    }

    new_cell_source = _update_parameters(parameter_cell_source, new_parameters)
    assert new_cell_source == '''
# Integer
TOP_K = 1
# Float
LEARNING_RATE = 0.1
# String
MOVIELENS_DATA_SIZE = "1m"
# List
RANKING_METRICS = ['ndcg_at_k', 'precision_at_k']
# Boolean
EVALUATE_WHILE_TRAINING = False
'''


@pytest.mark.notebooks
def test_notebook_execution(notebook_programmatic, output_notebook, kernel_name):
    """Test that the notebook executes and returns the correct results without params."""
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
    )

    results = read_notebook(output_notebook)
    assert results["response1"] == 3
    assert results["response2"] is True
    assert results["response3"] == 7


@pytest.mark.notebooks
@pytest.mark.parametrize(
    "parameters,expected_key,expected_value", [
        (dict(a=6), "response1", 8),            # Test the correct results with integers
        (dict(a=1.5), "response1", 3.5),        # Test the correct results with floats
        (dict(b="M"), "response2", True),       # Test the correct results with strings
        (dict(b="A"), "response2", "A"),        # Test the correct results with different strings
        (dict(b="100k"), "response2", "100k"),  # Test the correct results with strings that have numbers
        (dict(c=10), "response3", 12),          # Test the correct results with integers and a comment
    ]
)
def test_notebook_execution_with_parameters(
    notebook_programmatic,
    output_notebook,
    kernel_name,
    parameters,
    expected_key,
    expected_value,    
):
    """Test that the notebook executes."""
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
        parameters=parameters,
    )

    results = read_notebook(output_notebook)
    assert results[expected_key] == expected_value


@pytest.mark.notebooks
def test_notebook_execution_value_error_fails(
    notebook_programmatic, output_notebook, kernel_name
):
    """Test that the notebook fails with a value error."""
    with pytest.raises(nbclient.exceptions.CellExecutionError):
        execute_notebook(
            notebook_programmatic,
            output_notebook,
            kernel_name=kernel_name,
            parameters=dict(b=1),
        )
