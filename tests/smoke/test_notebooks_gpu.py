# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
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

    # There is too much variability to do an approx equal, just adding top values
    assert results["map"] < 0.05
    assert results["ndcg"] < 0.20
    assert results["precision"] < 0.17
    assert results["recall"] < 0.10


@pytest.mark.notebooks
@pytest.mark.gpu
def test_ncf_deep_dive(notebooks):
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
    assert results["map"] < 0.05
    assert results["ndcg"] < 0.35
    assert results["precision"] < 0.17
    assert results["recall"] < 0.1
    assert results["map2"] < 0.05
    assert results["ndcg2"] < 0.35
    assert results["precision2"] < 0.17
    assert results["recall2"] < 0.1


@pytest.mark.smoke
@pytest.mark.gpu
def test_fastai(notebooks):
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
@pytest.mark.deeprec
def test_notebook_xdeepfm(notebooks):
    notebook_path = notebooks["xdeepfm_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            EPOCHS_FOR_SYNTHETIC_RUN=20,
            EPOCHS_FOR_CRITEO_RUN=1,
            BATCH_SIZE_SYNTHETIC=128,
            BATCH_SIZE_CRITEO=512,
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["res_syn"]["auc"] == pytest.approx(0.982, rel=TOL, abs=ABS_TOL)
    assert results["res_syn"]["logloss"] == pytest.approx(0.2306, rel=TOL, abs=ABS_TOL)
    assert results["res_real"]["auc"] == pytest.approx(0.628, rel=TOL, abs=ABS_TOL)
    assert results["res_real"]["logloss"] == pytest.approx(0.5589, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_notebook_dkn(notebooks):
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
def test_wide_deep(notebooks, tmp):
    notebook_path = notebooks["wide_deep"]

    params = {
        "MOVIELENS_DATA_SIZE": "100k",
        "EPOCHS": 1,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": tmp,
        "EXPORT_DIR_BASE": tmp,
        "RATING_METRICS": ["rmse", "mae"],
        "RANKING_METRICS": ["ndcg_at_k", "precision_at_k"],
    }
    pm.execute_notebook(
        notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME, parameters=params
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    # Model performance is highly dependant on the initial random weights
    # when epochs is small with a small dataset.
    # Therefore, in the smoke-test context, rather check if the model training is working
    # with minimum performance metrics as follows:
    assert results["rmse"] < 2.0
    assert results["mae"] < 2.0
    assert results["ndcg_at_k"] > 0.0
    assert results["precision_at_k"] > 0.0
