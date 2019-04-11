# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import shutil
import subprocess
import sys

from reco_utils.nni.nni_utils import check_stopped
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME

TOL = 0.05
ABS_TOL = 0.05


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        (
                "1m",
                {
                    "map": 0.064012679,
                    "ndcg": 0.308012195,
                    "precision": 0.277214771,
                    "recall": 0.109291553,
                },
        ),
        (
                "10m",
                {
                    "map": 0.101402,
                    "ndcg": 0.321073,
                    "precision": 0.275766,
                    "recall": 0.156483,
                },
        ),
    ],
)
def test_sar_single_node_integration(notebooks, size, expected_values):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE=size),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        (
                "1m",
                {
                    "map": 0.033914,
                    "ndcg": 0.231570,
                    "precision": 0.211923,
                    "recall": 0.064663,
                },
        ),
        # ("10m", {"map": , "ndcg": , "precision": , "recall": }), # OOM on test machine
    ],
)
def test_baseline_deep_dive_integration(notebooks, size, expected_values):
    notebook_path = notebooks["baseline_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE=size),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        (
                "1m",
                dict(
                    rmse=0.89,
                    mae=0.70,
                    rsquared=0.36,
                    exp_var=0.36,
                    map=0.011,
                    ndcg=0.10,
                    precision=0.093,
                    recall=0.025,
                ),
        ),
        # 10m works but takes too long
    ],
)
def test_surprise_svd_integration(notebooks, size, expected_values):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE=size),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        (
            "1m",
            dict(
                rmse=0.959885,
                mae=0.690133,
                rsquared=0.264014,
                exp_var=0.264417,
                map=0.004857,
                ndcg=0.055128,
                precision=0.061142,
                recall=0.017789,
            ),
        )
    ],
)
def test_vw_deep_dive_integration(notebooks, size, expected_values):
    notebook_path = notebooks["vowpal_wabbit_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE=size, TOP_K=10),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
def test_nni_tuning_svd(notebooks, tmp):
    notebook_path = notebooks["nni_tuning_svd"]
    # First stop NNI in case it is running
    subprocess.run([sys.prefix + '/bin/nnictl', 'stop'])
    check_stopped()
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME,
                        parameters=dict(MOVIELENS_DATA_SIZE="100k",
                                        SURPRISE_READER="ml-100k",
                                        TMP_DIR=tmp,
                                        MAX_TRIAL_NUM=1,
                                        NUM_EPOCHS=1))
    # Clean up logs, saved models etc. under the NNI path
    nni_path = pm.read_notebook(OUTPUT_NOTEBOOK).data["nni_path"]
    shutil.rmtree(nni_path, ignore_errors=True)
