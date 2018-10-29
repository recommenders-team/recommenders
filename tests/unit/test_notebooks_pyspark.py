import os
import pytest
import pandas as pd
import papermill as pm
from tests.unit.notebooks_common import path_notebooks, OUTPUT_NOTEBOOK, KERNEL_NAME

notebooks = [
    (
        "sar_pyspark",
        os.path.join(path_notebooks(), "00_quick_start", "sar_pyspark_movielens.ipynb"),
    ),
    (
        "sarplus_movielens",
        os.path.join(path_notebooks(), "00_quick_start", "sarplus_movielens.ipynb"),
    ),
    ("data_split", os.path.join(path_notebooks(), "01_data", "data_split.ipynb")),
    (
        "sar_deep_dive",
        os.path.join(path_notebooks(), "02_modeling", "sar_deep_dive.ipynb"),
    ),
    (
        "als_deep_dive",
        os.path.join(path_notebooks(), "02_modeling", "als_deep_dive.ipynb"),
    ),
    ("evaluation", os.path.join(path_notebooks(), "03_evaluate", "evaluation.ipynb")),
]


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.parametrize("name,notebook_path", notebooks)
def test_sar_single_node_runs(name, notebook_path):
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
