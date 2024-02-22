# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest

from recommenders.utils.notebook_utils import execute_notebook, read_notebook


TOL = 0.05
ABS_TOL = 0.05


@pytest.mark.notebooks
def test_sar_single_node_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sar_single_node"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    results = read_notebook(output_notebook)

    assert results["map"] == pytest.approx(0.110591, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.382461, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.330753, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.176385, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
def test_baseline_deep_dive_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["baseline_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    results = read_notebook(output_notebook)

    assert results["rmse"] == pytest.approx(1.054252, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.846033, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.136435, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.136446, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.052850, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.248061, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.223754, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.108826, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
def test_surprise_svd_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = read_notebook(output_notebook)

    assert results["rmse"] == pytest.approx(0.96, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.75, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.29, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.29, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.013, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.1, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.095, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.032, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.skip(reason="VW pip package has installation incompatibilities")
def test_vw_deep_dive_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["vowpal_wabbit_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = read_notebook(output_notebook)

    assert results["rmse"] == pytest.approx(0.985920, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.71292, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.231199, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.231337, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.012535, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.096594, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.097770, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.037612, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
def test_lightgbm_quickstart_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["lightgbm_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MAX_LEAF=64,
            MIN_DATA=20,
            NUM_OF_TREES=100,
            TREE_LEARNING_RATE=0.15,
            EARLY_STOPPING_ROUNDS=20,
            METRIC="auc",
        ),
    )
    results = read_notebook(output_notebook)

    assert results["auc_basic"] == pytest.approx(0.7674, rel=TOL, abs=ABS_TOL)
    assert results["logloss_basic"] == pytest.approx(0.4669, rel=TOL, abs=ABS_TOL)
    assert results["auc_opt"] == pytest.approx(0.7757, rel=TOL, abs=ABS_TOL)
    assert results["logloss_opt"] == pytest.approx(0.4607, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
def test_cornac_bpr_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["cornac_bpr_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = read_notebook(output_notebook)

    assert results["map"] == pytest.approx(0.1091, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.4034, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.3550, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.1802, rel=TOL, abs=ABS_TOL)
