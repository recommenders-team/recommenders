# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest

from reco_utils.common.constants import SEED
from reco_utils.common.gpu_utils import get_number_gpus
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.5
ABS_TOL = 0.05


@pytest.mark.smoke
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.smoke
@pytest.mark.gpu
def test_ncf_smoke(notebooks):
    notebook_path = notebooks["ncf"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=256),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["map"] == pytest.approx(0.0409234, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.1773, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.160127, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0879193, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
@pytest.mark.gpu
def test_ncf_deep_dive_smoke(notebooks):
    notebook_path = notebooks["ncf_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=1024
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    # There is too much variability to do an approx equal, just adding top values
    assert results["map"] == pytest.approx(0.0370396, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.29423, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.144539, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0730272, rel=TOL, abs=ABS_TOL)
    assert results["map2"] == pytest.approx(0.028952, rel=TOL, abs=ABS_TOL)
    assert results["ndcg2"] == pytest.approx(0.143744, rel=TOL, abs=ABS_TOL)
    assert results["precision2"] == pytest.approx(0.127041, rel=TOL, abs=ABS_TOL)
    assert results["recall2"] == pytest.approx(0.0584491, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
@pytest.mark.gpu
def test_fastai_smoke(notebooks):
    notebook_path = notebooks["fastai"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(0.959352, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.766504, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.287902, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.289008, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.024379, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.148380, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.138494, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.058747, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
@pytest.mark.gpu
def test_xdeepfm_smoke(notebooks):
    notebook_path = notebooks["xdeepfm_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            EPOCHS_FOR_SYNTHETIC_RUN=1,
            EPOCHS_FOR_CRITEO_RUN=1,
            BATCH_SIZE_SYNTHETIC=128,
            BATCH_SIZE_CRITEO=512,
            RANDOM_SEED=SEED,
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["res_syn"]["auc"] == pytest.approx(0.5043, rel=TOL, abs=ABS_TOL)
    assert results["res_syn"]["logloss"] == pytest.approx(0.7046, rel=TOL, abs=ABS_TOL)
    assert results["res_real"]["auc"] == pytest.approx(0.7251, rel=TOL, abs=ABS_TOL)
    assert results["res_real"]["logloss"] == pytest.approx(0.508, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
@pytest.mark.gpu
def test_dkn_smoke(notebooks):
    notebook_path = notebooks["dkn_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(epoch=1),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["res"]["auc"] == pytest.approx(0.4707, rel=TOL, abs=ABS_TOL)
    assert results["res"]["acc"] == pytest.approx(0.5725, rel=TOL, abs=ABS_TOL)
    # assert results["res"]["f1"] == pytest.approx(0.7281, rel=TOL, abs=ABS_TOL) # FIXME: issue #528


@pytest.mark.smoke
@pytest.mark.gpu
def test_wide_deep_smoke(notebooks, tmp):
    notebook_path = notebooks["wide_deep"]

    params = {
        "MOVIELENS_DATA_SIZE": "100k",
        "EPOCHS": 1,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": tmp,
        "EXPORT_DIR_BASE": tmp,
        "RATING_METRICS": ["rmse", "mae"],
        "RANKING_METRICS": ["ndcg_at_k", "precision_at_k"],
        "RANDOM_SEED": SEED,
    }
    pm.execute_notebook(
        notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME, parameters=params
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(1.0394, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.836116, rel=TOL, abs=ABS_TOL)
    assert results["ndcg_at_k"] == pytest.approx(0.0954757, rel=TOL, abs=ABS_TOL)
    assert results["precision_at_k"] == pytest.approx(0.080912, rel=TOL, abs=ABS_TOL)
