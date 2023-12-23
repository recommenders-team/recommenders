# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pytest

from recommenders.utils.gpu_utils import get_number_gpus
from recommenders.utils.notebook_utils import execute_notebook


@pytest.mark.notebooks
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.notebooks
@pytest.mark.gpu
def test_fastai(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["fastai"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="mock100", EPOCHS=1),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_ncf(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["ncf"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE="mock100", EPOCHS=1, BATCH_SIZE=1024
        ),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_ncf_deep_dive(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["ncf_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE="mock100", EPOCHS=1, BATCH_SIZE=2048
        ),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_xdeepfm(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["xdeepfm_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            EPOCHS=1,
            BATCH_SIZE=1024,
        ),
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_wide_deep(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["wide_deep"]

    # Simple test (train only 1 batch == 1 step)
    model_dir = os.path.join(tmp, "wide_deep_0")
    os.mkdir(model_dir)
    params = {
        "MOVIELENS_DATA_SIZE": "mock100",
        "STEPS": 1,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": model_dir,
        "EXPORT_DIR_BASE": model_dir,
        "RATING_METRICS": ["rmse"],
        "RANKING_METRICS": ["ndcg_at_k"],
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )

    # Test with different parameters
    model_dir = os.path.join(tmp, "wide_deep_1")
    os.mkdir(model_dir)
    params = {
        "MOVIELENS_DATA_SIZE": "mock100",
        "STEPS": 1,
        "ITEM_FEAT_COL": None,
        "EVALUATE_WHILE_TRAINING": True,
        "MODEL_DIR": model_dir,
        "EXPORT_DIR_BASE": model_dir,
        "RATING_METRICS": ["rsquared"],
        "RANKING_METRICS": ["map_at_k"],
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )


@pytest.mark.notebooks
@pytest.mark.gpu
def test_dkn_quickstart(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["dkn_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(EPOCHS=1, BATCH_SIZE=500, HISTORY_SIZE=5),
    )
