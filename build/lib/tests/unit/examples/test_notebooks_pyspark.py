# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import sys
import pytest

from recommenders.utils.constants import (
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)
from recommenders.utils.notebook_utils import execute_notebook


# This is a flaky test that can fail unexpectedly
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 1087.56s in Windows, while in Linux 52.51s"
)
def test_als_pyspark_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["als_pyspark"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MOVIELENS_DATA_SIZE="mock100",
            COL_USER=DEFAULT_USER_COL,
            COL_ITEM=DEFAULT_ITEM_COL,
            COL_RATING=DEFAULT_RATING_COL,
        ),
    )


@pytest.mark.notebooks
@pytest.mark.spark
def test_data_split_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["data_split"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


# This is a flaky test that can fail unexpectedly
@pytest.mark.flaky(reruns=5, reruns_delay=3)
@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 2764.50s in Windows, while in Linux 124.35s"
)
def test_als_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["als_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MOVIELENS_DATA_SIZE="mock100",
            COL_USER=DEFAULT_USER_COL,
            COL_ITEM=DEFAULT_ITEM_COL,
            COL_RATING=DEFAULT_RATING_COL,
        ),
    )


# This is a flaky test that can fail unexpectedly
@pytest.mark.flaky(reruns=5, reruns_delay=3)
@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 583.75s in Windows, while in Linux 71.77s"
)
def test_evaluation_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["evaluation"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


# This is a flaky test that can fail unexpectedly
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.notebooks
@pytest.mark.spark
def test_evaluation_diversity_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["evaluation_diversity"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10,
            MOVIELENS_DATA_SIZE="mock100",
            COL_USER=DEFAULT_USER_COL,
            COL_ITEM=DEFAULT_ITEM_COL,
            COL_RATING=DEFAULT_RATING_COL,
        ),
    )


# mock100 dataset throws the following error:
#   TrainValidationSplit IllegalArgumentException: requirement failed:
#   Nothing has been added to this summarizer.
# This seems to be caused by cold start problem -- https://stackoverflow.com/questions/58827795/requirement-failed-nothing-has-been-added-to-this-summarizer
# In terms of the processing speed at Spark, "100k" dataset does not take much longer than "mock100" dataset and thus use "100k" here to go around the issue. 
@pytest.mark.flaky(reruns=5, reruns_delay=2)
@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(
    sys.platform == "win32", reason="Takes 2409.69s in Windows, while in Linux 138.30s"
)
def test_spark_tuning(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["spark_tuning"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MOVIELENS_DATA_SIZE="100k",  # Note: mock100 throws an error   
            NUMBER_CORES="1",
            NUMBER_ITERATIONS=3,
            SUBSET_RATIO=0.5,
            RANK=[5, 10],
            REG=[0.1, 0.01],
        ),
    )


@pytest.mark.notebooks
@pytest.mark.spark
@pytest.mark.skipif(sys.platform == "win32", reason="Not implemented on Windows")
def test_mmlspark_lightgbm_criteo_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["mmlspark_lightgbm_criteo"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(DATA_SIZE="sample", NUM_ITERATIONS=10),
    )
