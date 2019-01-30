# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.common.gpu_utils import get_number_gpus
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
import papermill as pm


@pytest.mark.notebooks
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.notebooks
@pytest.mark.gpu
def test_fastai(notebooks):
    notebook_path = notebooks["fastai"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_ncf(notebooks):
    notebook_path = notebooks["ncf"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=1024
        ),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
@pytest.mark.skip(
    reason="as of now, it takes too long to do a unit test, see issue #466"
)
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

