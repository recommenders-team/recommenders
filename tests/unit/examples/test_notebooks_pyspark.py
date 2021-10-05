# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest
try:
    import papermill as pm
except ImportError:
    pass  # disable error while collecting tests for non-notebook environments

from recommenders.utils.constants import DEFAULT_RATING_COL, DEFAULT_USER_COL, DEFAULT_ITEM_COL


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 1087.56s in Windows, while in Linux 52.51s"
)
def test_als_pyspark_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["als_pyspark"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
@pytest.mark.spark
def test_data_split_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["data_split"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 2764.50s in Windows, while in Linux 124.35s"
)
def test_als_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["als_deep_dive"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name,
                        parameters=dict(
                            MOVIELENS_DATA_SIZE="mock100",
                            COL_USER=DEFAULT_USER_COL,
                            COL_ITEM=DEFAULT_ITEM_COL,
                            COL_RATING=DEFAULT_RATING_COL,
                        ))


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 583.75s in Windows, while in Linux 71.77s"
)
def test_evaluation_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["evaluation"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
@pytest.mark.spark
def test_evaluation_diversity_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["evaluation_diversity"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name,
                        parameters=dict(
                            TOP_K=10,
                            MOVIELENS_DATA_SIZE="mock100",
                            COL_USER=DEFAULT_USER_COL,
                            COL_ITEM=DEFAULT_ITEM_COL,
                            COL_RATING=DEFAULT_RATING_COL,
                        ))


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 2409.69s in Windows, while in Linux 138.30s"
)
def test_spark_tuning(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["spark_tuning"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MOVIELENS_DATA_SIZE="mock100",
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
def test_mmlspark_lightgbm_criteo_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["mmlspark_lightgbm_criteo"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(DATA_SIZE="sample", NUM_ITERATIONS=10, EARLY_STOPPING_ROUND=2),
    )
