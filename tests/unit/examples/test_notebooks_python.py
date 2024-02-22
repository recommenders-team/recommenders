# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import sys
import pytest

import recommenders
from recommenders.utils.notebook_utils import execute_notebook, read_notebook


TOL = 0.05
ABS_TOL = 0.05


@pytest.mark.notebooks
def test_template_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["template"]
    execute_notebook(
        notebook_path,
        output_notebook,
        parameters=dict(RECOMMENDERS_VERSION=recommenders.__version__),
        kernel_name=kernel_name,
    )
    results = read_notebook(output_notebook)

    assert len(results) == 1
    assert results["checked_version"]


@pytest.mark.notebooks
def test_sar_single_node_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sar_single_node"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
def test_sar_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sar_deep_dive"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
def test_baseline_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["baseline_deep_dive"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
def test_surprise_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE="mock100"),
    )


@pytest.mark.notebooks
def test_lightgbm(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["lightgbm_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MAX_LEAF=32,
            MIN_DATA=20,
            NUM_OF_TREES=10,
            TREE_LEARNING_RATE=0.15,
            EARLY_STOPPING_ROUNDS=20,
            METRIC="auc",
        ),
    )


@pytest.mark.notebooks
def test_cornac_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["cornac_bpr_deep_dive"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


@pytest.mark.notebooks
@pytest.mark.experimental
@pytest.mark.skip(reason="rlrmc doesn't work with any officially released pymanopt package")
def test_rlrmc_quickstart_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["rlrmc_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(rank_parameter=2, MOVIELENS_DATA_SIZE="mock100"),
    )


@pytest.mark.notebooks
@pytest.mark.experimental
@pytest.mark.skip(reason="VW pip package has installation incompatibilities")
def test_vw_deep_dive_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["vowpal_wabbit_deep_dive"]
    execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)
