# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 1087.56s in Windows, while in Linux 52.51s"
)
def test_als_pyspark_runs(notebooks):
    notebook_path = notebooks["als_pyspark"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
def test_data_split_runs(notebooks):
    notebook_path = notebooks["data_split"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 2764.50s in Windows, while in Linux 124.35s"
)
def test_als_deep_dive_runs(notebooks):
    notebook_path = notebooks["als_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 583.75s in Windows, while in Linux 71.77s"
)
def test_evaluation_runs(notebooks):
    notebook_path = notebooks["evaluation"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 2409.69s in Windows, while in Linux 138.30s"
)
def test_spark_tuning(notebooks):
    notebook_path = notebooks["spark_tuning"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            NUMBER_CORES="*",
            NUMBER_ITERATIONS=3,
            SUBSET_RATIO=0.5,
            RANK=[5, 5],
            REG=[0.1, 0.01],
        ),
    )


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(sys.platform == "win32", reason="Not implemented on Windows")
def test_mmlspark_lightgbm_criteo_runs(notebooks):
    notebook_path = notebooks["mmlspark_lightgbm_criteo"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(DATA_SIZE="sample", NUM_ITERATIONS=10, EARLY_STOPPING_ROUND=2),
    )
