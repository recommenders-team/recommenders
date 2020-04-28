# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.05
ABS_TOL = 0.05


@pytest.mark.smoke
def test_sar_single_node_smoke(notebooks):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["map"] == pytest.approx(0.110591, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.382461, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.330753, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.176385, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
def test_baseline_deep_dive_smoke(notebooks):
    notebook_path = notebooks["baseline_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(1.054252, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.846033, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.136435, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.136446, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.052850, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.248061, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.223754, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.108826, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
def test_surprise_svd_smoke(notebooks):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(0.96, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.75, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.29, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.29, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.013, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.1, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.095, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.032, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
def test_vw_deep_dive_smoke(notebooks):
    notebook_path = notebooks["vowpal_wabbit_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(0.985920, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.71292, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.231199, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.231337, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.012535, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.096594, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.097770, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.037612, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
def test_lightgbm_quickstart_smoke(notebooks):
    notebook_path = notebooks["lightgbm_quickstart"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            MAX_LEAF=64,
            MIN_DATA=20,
            NUM_OF_TREES=100,
            TREE_LEARNING_RATE=0.15,
            EARLY_STOPPING_ROUNDS=20,
            METRIC="auc",
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["res_basic"]["auc"] == pytest.approx(0.7674, rel=TOL, abs=ABS_TOL)
    assert results["res_basic"]["logloss"] == pytest.approx(
        0.4669, rel=TOL, abs=ABS_TOL
    )
    assert results["res_optim"]["auc"] == pytest.approx(0.7757, rel=TOL, abs=ABS_TOL)
    assert results["res_optim"]["logloss"] == pytest.approx(
        0.4607, rel=TOL, abs=ABS_TOL
    )


@pytest.mark.smoke
def test_cornac_bpr_smoke(notebooks):
    notebook_path = notebooks["cornac_bpr_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["map"] == pytest.approx(0.1091, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.4034, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.3550, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.1802, rel=TOL, abs=ABS_TOL)
