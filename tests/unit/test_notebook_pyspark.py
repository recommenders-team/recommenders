import os
import pytest
import pandas as pd
import papermill as pm


OUTPUT_NOTEBOOK = "output.ipynb"


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, "notebooks"
        )
    )
    paths = {
        "sar_pyspark": os.path.join(
            folder_notebooks, "00_quick_start", "sar_pyspark_movielens.ipynb"
        )
    }
    return paths


@pytest.mark.notebooks
@pytest.mark.spark
def test_sar_single_node_runs(notebooks):
    notebook_path = notebooks["sar_pyspark"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK)

