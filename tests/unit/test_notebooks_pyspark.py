# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
from tests.unit.notebooks_common import path_notebooks, OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "als_pyspark": os.path.join(
            folder_notebooks, "00_quick_start", "als_pyspark_movielens.ipynb"
        ),
        "sar_pyspark": os.path.join(
            folder_notebooks, "00_quick_start", "sar_pyspark_movielens.ipynb"
        ),
        "data_split": os.path.join(folder_notebooks, "01_data", "data_split.ipynb"),
        "sar_deep_dive": os.path.join(
            folder_notebooks, "02_modeling", "sar_deep_dive.ipynb"
        ),
        "als_deep_dive": os.path.join(
            folder_notebooks, "02_modeling", "als_deep_dive.ipynb"
        ),
        "evaluation": os.path.join(folder_notebooks, "03_evaluate", "evaluation.ipynb"),
    }
    return paths


@pytest.mark.notebooks
@pytest.mark.spark
def test_als_pyspark_runs(notebooks):
    notebook_path = notebooks["als_pyspark"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
def test_sar_pyspark_runs(notebooks):
    notebook_path = notebooks["sar_pyspark"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
def test_data_split_runs(notebooks):
    notebook_path = notebooks["data_split"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
def test_sar_deep_dive_runs(notebooks):
    notebook_path = notebooks["sar_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
def test_als_deep_dive_runs(notebooks):
    notebook_path = notebooks["als_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
def test_evaluation_runs(notebooks):
    notebook_path = notebooks["evaluation"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
