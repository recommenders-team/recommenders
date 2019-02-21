# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import shutil
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
def test_ncf_deep_dive(notebooks):
    notebook_path = notebooks["ncf_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=2048
        ),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_wide_deep(notebooks):
    notebook_path = notebooks["wide_deep"]

    MODEL_DIR = 'model_checkpoints'
    params = {
        'MOVIELENS_DATA_SIZE': '100k',
        'EPOCHS': 1,
        'EVALUATE_WHILE_TRAINING': False,
        'MODEL_DIR': MODEL_DIR,
        'EXPORT_DIR_BASE': MODEL_DIR,
        'RATING_METRICS': ['rmse', 'mae'],
        'RANKING_METRICS': ['ndcg_at_k', 'precision_at_k'],
    }

    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=params,
    )

    shutil.rmtree(MODEL_DIR, ignore_errors=True)
