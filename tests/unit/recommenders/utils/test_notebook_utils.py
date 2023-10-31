# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import nbclient
import pytest
import papermill as pm
import scrapbook as sb
from pathlib import Path

from recommenders.utils.notebook_utils import (
    is_jupyter,
    is_databricks,
    execute_notebook,
    read_notebook,
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
    pm.execute_notebook(
        notebook_types,
        output_notebook,
        kernel_name=kernel_name,
    )
    nb = sb.read_notebook(output_notebook)
    df = nb.scraps.dataframe
    result_is_jupyter = df.loc[df["name"] == "is_jupyter", "data"].values[0]
    assert result_is_jupyter  # is True not allowed
    result_is_databricks = df.loc[df["name"] == "is_databricks", "data"].values[0]
    assert not result_is_databricks


@pytest.mark.spark
@pytest.mark.notebooks
@pytest.mark.skip(reason="TODO: Implement this")
def test_is_databricks():
    pass


def test_notebook_execution_int(notebook_programmatic, output_notebook, kernel_name):
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(a=6),
    )

    results = read_notebook(output_notebook)
    assert results["response1"] == 8


def test_notebook_execution_float(notebook_programmatic, output_notebook, kernel_name):
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(a=1.5),
    )

    results = read_notebook(output_notebook)
    assert results["response1"] == 3.5


def test_notebook_execution_letter(notebook_programmatic, output_notebook, kernel_name):
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(b="M"),
    )

    results = read_notebook(output_notebook)
    assert results["response2"] is True


def test_notebook_execution_other_letter(
    notebook_programmatic, output_notebook, kernel_name
):
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(b="A"),
    )

    results = read_notebook(output_notebook)
    assert results["response2"] == "A"


def test_notebook_execution_value_error_fails(
    notebook_programmatic, output_notebook, kernel_name
):
    with pytest.raises(nbclient.exceptions.CellExecutionError):
        execute_notebook(
            notebook_programmatic,
            output_notebook,
            kernel_name=kernel_name,
            parameters=dict(b=1),
        )


def test_notebook_execution_int_with_comment(
    notebook_programmatic, output_notebook, kernel_name
):
    execute_notebook(
        notebook_programmatic,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(c=10),
    )

    results = read_notebook(output_notebook)
    assert results["response3"] == 12
