# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
import sys

from reco_utils.tuning.nni.nni_utils import check_experiment_status, NNI_STATUS_URL
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
                "map": 0.060579,
                "ndcg": 0.299245,
                "precision": 0.270116,
                "recall": 0.104350,
            },
        ),
        (
            "10m",
            {
                "map": 0.098745,
                "ndcg": 0.319625,
                "precision": 0.275756,
                "recall": 0.154014,
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
@pytest.mark.skipif(sys.platform == "win32", reason="nni not installable on windows")
def test_nni_tuning_svd(notebooks, tmp):
    notebook_path = notebooks["nni_tuning_svd"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            MOVIELENS_DATA_SIZE="100k",
            SURPRISE_READER="ml-100k",
            TMP_DIR=tmp,
            MAX_TRIAL_NUM=1,
            NUM_EPOCHS=1,
            WAITING_TIME=20,
            MAX_RETRIES=50,
        ),
    )


@pytest.mark.integration
def test_wikidata_integration(notebooks, tmp):
    notebook_path = notebooks["wikidata_knowledge_graph"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            MOVIELENS_DATA_SIZE="100k", MOVIELENS_SAMPLE=True, MOVIELENS_SAMPLE_SIZE=5
        ),
    )

    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]
    # NOTE: The return number should be always 5, but sometimes we get less because wikidata is unstable
    assert results["length_result"] >= 1

@pytest.mark.integration
def test_mind_utils_integration(notebooks, tmp):
    notebook_path = notebooks["mind_utils"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(
            mind_type="small", word_embedding_dim=300
        ),
    )

    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]
    assert results["utils_state"]["vert_num"] == 17
    assert results["utils_state"]["subvert_num"] == 17
    assert results["utils_state"]["word_num"] == 31029
    assert results["utils_state"]["word_num_all"] == 55028
    assert results["utils_state"]["embedding_exist_num"] == 29081
    assert results["utils_state"]["embedding_exist_num_all"] == 48422
    assert results["utils_state"]["uid2index"] == 50000

@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        ("1m", dict(map=0.081390, ndcg=0.406627, precision=0.373228, recall=0.132444)),
        # 10m works but takes too long
    ],
)
def test_cornac_bpr_integration(notebooks, size, expected_values):
    notebook_path = notebooks["cornac_bpr_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE=size),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


def test_xlearn_fm_integration(notebooks):
    notebook_path = notebooks["xlearn_fm_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(LEARNING_RATE=0.2, EPOCH=10),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["auc_score"] == pytest.approx(0.75, rel=TOL, abs=ABS_TOL)


@pytest.mark.integration
@pytest.mark.parametrize(
    "expected_values",
    [
        (
            {
                "rmse": 0.4969,
                "mae": 0.4761
            }
        )
    ],
)
def test_geoimc_integration(notebooks, expected_values):
    notebook_path = notebooks["geoimc_quickstart"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)
