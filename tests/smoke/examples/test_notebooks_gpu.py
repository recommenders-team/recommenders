# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest

from recommenders.utils.gpu_utils import get_number_gpus
from recommenders.utils.notebook_utils import execute_notebook, read_notebook


TOL = 0.5
ABS_TOL = 0.05


@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.notebooks
@pytest.mark.gpu
def test_ncf_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["ncf"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=256),
    )
    results = read_notebook(output_notebook)

    assert results["map"] == pytest.approx(0.0409234, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.1773, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.160127, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0879193, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_ncf_deep_dive_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["ncf_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=1024),
    )
    results = read_notebook(output_notebook)

    # There is too much variability to do an approx equal, just adding top values
    assert results["map"] == pytest.approx(0.0370396, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.29423, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.144539, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0730272, rel=TOL, abs=ABS_TOL)
    assert results["map2"] == pytest.approx(0.028952, rel=TOL, abs=ABS_TOL)
    assert results["ndcg2"] == pytest.approx(0.143744, rel=TOL, abs=ABS_TOL)
    assert results["precision2"] == pytest.approx(0.127041, rel=TOL, abs=ABS_TOL)
    assert results["recall2"] == pytest.approx(0.0584491, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_fastai_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["fastai"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1),
    )
    results = read_notebook(output_notebook)

    assert results["rmse"] == pytest.approx(0.959352, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.766504, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.287902, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.289008, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.024379, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.148380, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.138494, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.058747, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_xdeepfm_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["xdeepfm_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            EPOCHS=1,
            BATCH_SIZE=512,
            RANDOM_SEED=42,
        ),
    )
    results = read_notebook(output_notebook)

    assert results["auc"] == pytest.approx(0.7251, rel=TOL, abs=ABS_TOL)
    assert results["logloss"] == pytest.approx(0.508, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_wide_deep_smoke(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["wide_deep"]

    params = {
        "MOVIELENS_DATA_SIZE": "100k",
        "STEPS": 1000,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": tmp,
        "EXPORT_DIR_BASE": tmp,
        "RATING_METRICS": ["rmse", "mae"],
        "RANKING_METRICS": ["ndcg_at_k", "precision_at_k"],
        "RANDOM_SEED": 42,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    assert results["rmse"] == pytest.approx(1.06034, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.876228, rel=TOL, abs=ABS_TOL)
    assert results["ndcg_at_k"] == pytest.approx(0.181513, rel=TOL, abs=ABS_TOL)
    assert results["precision_at_k"] == pytest.approx(0.158961, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_naml_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["naml_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, batch_size=64, seed=42, MIND_type="demo"),
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        0.5801, rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(0.2512, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_nrms_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["nrms_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, seed=42, MIND_type="demo"),
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        0.5768, rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(0.2457, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_npa_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["npa_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, batch_size=64, seed=42, MIND_type="demo"),
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        0.5861, rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(0.255, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_lstur_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["lstur_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, batch_size=64, seed=40, MIND_type="demo"),
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        0.5977, rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(0.2618, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
def test_cornac_bivae_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["cornac_bivae_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = read_notebook(output_notebook)

    assert results["map"] == pytest.approx(0.146552, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.474124, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.412527, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.225064, rel=TOL, abs=ABS_TOL)
