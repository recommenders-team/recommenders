# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
from reco_utils.common.gpu_utils import get_number_gpus
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

TOL = 0.5
ABS_TOL = 0.05


@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "size, epochs, expected_values",
    [
        (
            "1m",
            10,
            {
                "map": 0.024821,
                "ndcg": 0.153396,
                "precision": 0.143046,
                "recall": 0.056590,
            },
        ),
        # ("10m", 5, {"map": 0.024821, "ndcg": 0.153396, "precision": 0.143046, "recall": 0.056590})# takes too long
    ],
)
def test_ncf_integration(notebooks, size, epochs, expected_values):
    notebook_path = notebooks["ncf"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE=size, EPOCHS=epochs, BATCH_SIZE=512
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "size, epochs, batch_size, expected_values",
    [
        (
            "100k",
            10,
            512,
            {
                "map": 0.045746,
                "ndcg": 0.3739307,
                "precision": 0.183987,
                "recall": 0.105546,
                "map2": 0.049723,
                "ndcg2": 0.201361,
                "precision2": 0.180276,
                "recall2": 0.103631,
            },
        )
    ],
)
def test_ncf_deep_dive_integration(
    notebooks, size, epochs, batch_size, expected_values
):
    notebook_path = notebooks["ncf_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE=size, EPOCHS=epochs, BATCH_SIZE=batch_size
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "size, epochs, expected_values",
    [
        (
            "1m",
            10,
            {
                "map": 0.025739,
                "ndcg": 0.183417,
                "precision": 0.167246,
                "recall": 0.054307,
                "rmse": 0.881267,
                "mae": 0.700747,
                "rsquared": 0.379963,
                "exp_var": 0.382842,
            },
        ),
        # ("10m", 5, ), # it gets an OOM on pred = learner.model.forward(u, m)
    ],
)
def test_fastai_integration(notebooks, size, epochs, expected_values):
    notebook_path = notebooks["fastai"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE=size, EPOCHS=epochs),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "syn_epochs, criteo_epochs, expected_values",
    [
        (
            15,
            30,
            {
                "res_syn": {
                    "auc": 0.9716,
                    "logloss": 0.2278,
                },
                "res_real": {
                    "auc": 0.749,
                    "logloss": 0.4926,
                },
            },
        )
    ],
)
def test_xdeepfm_integration(notebooks, syn_epochs, criteo_epochs, expected_values):
    notebook_path = notebooks["xdeepfm_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            EPOCHS_FOR_SYNTHETIC_RUN=syn_epochs,
            EPOCHS_FOR_CRITEO_RUN=criteo_epochs,
            RANDOM_SEED=SEED,
        ),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key]["auc"] == pytest.approx(value["auc"], rel=TOL, abs=ABS_TOL)
        assert results[key]["logloss"] == pytest.approx(value["logloss"], rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "size, epochs, expected_values",
    [
        (
            "1m",
            10,
            {
                "rmse": 0.911973,
                "mae": 0.71469,
                "rsquared": 0.335589,
                "exp_var": 0.336365,
                "ndcg_at_k": 0.0706927,
                "map_at_k": 0.00849144,
                "precision_at_k": 0.0571452,
                "recall_at_k": 0.0180048,
            },
        )
    ],
)
def test_wide_deep_integration(notebooks, size, epochs, expected_values, tmp):
    notebook_path = notebooks["wide_deep"]

    params = {
        "MOVIELENS_DATA_SIZE": size,
        "EPOCHS": epochs,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": tmp,
        "EXPORT_DIR_BASE": tmp,
        "RATING_METRICS": ["rmse", "mae", "rsquared", "exp_var"],
        "RANKING_METRICS": ["ndcg_at_k", "map_at_k", "precision_at_k", "recall_at_k"],
        "RANDOM_SEED": SEED,
    }
    pm.execute_notebook(
        notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME, parameters=params
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)
