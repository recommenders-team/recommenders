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
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, 
                        MOVIELENS_DATA_SIZE="100k",
                        EPOCHS=1,
                        BATCH_SIZE=1024),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["map"] == pytest.approx(0.040266, TOL)
    assert results["ndcg"] == pytest.approx(0.180131, TOL)
    assert results["precision"] == pytest.approx(0.161082, TOL)
    assert results["recall"] == pytest.approx(0.084300, TOL)


@pytest.mark.notebooks
@pytest.mark.gpu
@pytest.mark.skip(reason="as of now, it takes too long to do a smoke test")
def test_ncf_deep_dive(notebooks):
    notebook_path = notebooks["ncf_deep_dive"]
    pm.execute_notebook(notebook_path, 
                        OUTPUT_NOTEBOOK, 
                        kernel_name=KERNEL_NAME,
                        parameters=dict(TOP_K=10, 
                                        MOVIELENS_DATA_SIZE="100k",
                                        EPOCHS=1,
                                        BATCH_SIZE=1024),
                       )
  

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

