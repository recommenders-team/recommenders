# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
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
def test_xdeepfm(notebooks):
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
        ),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_wide_deep(notebooks, tmp):
    notebook_path = notebooks["wide_deep"]

    model_dir = os.path.join(tmp, "wide_deep_0")
    os.mkdir(model_dir)
    params = {
        'MOVIELENS_DATA_SIZE': '100k',
        'EPOCHS': 0,
        'EVALUATE_WHILE_TRAINING': False,
        'MODEL_DIR': model_dir,
        'EXPORT_DIR_BASE': model_dir,
        'RATING_METRICS': ['rmse'],
        'RANKING_METRICS': ['ndcg_at_k'],
    }
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=params,
    )

    # Test w/o item features
    model_dir = os.path.join(tmp, "wide_deep_1")
    os.mkdir(model_dir)
    params = {
        'MOVIELENS_DATA_SIZE': '100k',
        'EPOCHS': 0,
        'ITEM_FEAT_COL': None,
        'EVALUATE_WHILE_TRAINING': True,
        'MODEL_DIR': model_dir,
        'EXPORT_DIR_BASE': model_dir,
        'RATING_METRICS': ['rsquared'],
        'RANKING_METRICS': ['map_at_k'],
    }
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=params,
    )
