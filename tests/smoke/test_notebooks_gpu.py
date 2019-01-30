# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
from reco_utils.common.gpu_utils import get_number_gpus
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.05


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

    # There is a high variability on this algo, adjusting the tolerance
    tolerance = TOL*2
    assert results["map"] == pytest.approx(0.027992, tolerance)
    assert results["ndcg"] == pytest.approx(0.143840, tolerance)
    assert results["precision"] == pytest.approx(0.129374, tolerance)
    assert results["recall"] == pytest.approx(0.062546, tolerance)
    assert results["map2"] == pytest.approx(0.029929, tolerance)
    assert results["ndcg2"] == pytest.approx(0.146640, tolerance)
    assert results["precision2"] == pytest.approx(0.132238, tolerance)
    assert results["recall2"] == pytest.approx(0.063981, tolerance)

    
@pytest.mark.smoke
@pytest.mark.gpu
def test_fastai(notebooks):
    notebook_path = notebooks["fastai"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(0.959352, TOL)
    assert results["mae"] == pytest.approx(0.766504, TOL)
    assert results["rsquared"] == pytest.approx(0.287902, TOL)
    assert results["exp_var"] == pytest.approx(0.289008, TOL)
    assert results["map"] == pytest.approx(0.024379, TOL)
    assert results["ndcg"] == pytest.approx(0.148380, TOL)
    assert results["precision"] == pytest.approx(0.138494, TOL)
    assert results["recall"] == pytest.approx(0.058747, TOL)

