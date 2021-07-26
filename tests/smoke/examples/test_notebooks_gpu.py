# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pytest

try:
    import papermill as pm
    import scrapbook as sb
except ImportError:
    pass  # disable error while collecting tests for non-notebook environments

from recommenders.utils.gpu_utils import get_number_gpus


TOL = 0.5
ABS_TOL = 0.05


@pytest.mark.smoke
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_ncf_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["ncf"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=256),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["map"] == pytest.approx(0.0409234, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.1773, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.160127, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0879193, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_ncf_deep_dive_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["ncf_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1, BATCH_SIZE=1024
        ),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    # There is too much variability to do an approx equal, just adding top values
    assert results["map"] == pytest.approx(0.0370396, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.29423, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.144539, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0730272, rel=TOL, abs=ABS_TOL)
    assert results["map2"] == pytest.approx(0.028952, rel=TOL, abs=ABS_TOL)
    assert results["ndcg2"] == pytest.approx(0.143744, rel=TOL, abs=ABS_TOL)
    assert results["precision2"] == pytest.approx(0.127041, rel=TOL, abs=ABS_TOL)
    assert results["recall2"] == pytest.approx(0.0584491, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_fastai_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["fastai"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k", EPOCHS=1),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["rmse"] == pytest.approx(0.959352, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.766504, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.287902, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.289008, rel=TOL, abs=ABS_TOL)
    assert results["map"] == pytest.approx(0.024379, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.148380, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.138494, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.058747, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_xdeepfm_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["xdeepfm_quickstart"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            EPOCHS_FOR_SYNTHETIC_RUN=1,
            EPOCHS_FOR_CRITEO_RUN=1,
            BATCH_SIZE_SYNTHETIC=128,
            BATCH_SIZE_CRITEO=512,
            RANDOM_SEED=42,
        ),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["res_syn"]["auc"] == pytest.approx(0.5043, rel=TOL, abs=ABS_TOL)
    assert results["res_syn"]["logloss"] == pytest.approx(0.7046, rel=TOL, abs=ABS_TOL)
    assert results["res_real"]["auc"] == pytest.approx(0.7251, rel=TOL, abs=ABS_TOL)
    assert results["res_real"]["logloss"] == pytest.approx(0.508, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_wide_deep_smoke(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["wide_deep"]

    params = {
        "MOVIELENS_DATA_SIZE": "100k",
        "STEPS": 1000,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": tmp,
        "EXPORT_DIR_BASE": tmp,
        "RATING_METRICS": ["rmse", "mae"],
        "RANKING_METRICS": ["ndcg_at_k", "precision_at_k"],
        "RANDOM_SEED": 42,
    }
    pm.execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["rmse"] == pytest.approx(1.06034, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.876228, rel=TOL, abs=ABS_TOL)
    assert results["ndcg_at_k"] == pytest.approx(0.181513, rel=TOL, abs=ABS_TOL)
    assert results["precision_at_k"] == pytest.approx(0.158961, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_naml_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["naml_quickstart"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, seed=42, MIND_type="demo"),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["res_syn"]["group_auc"] == pytest.approx(
        0.5801, rel=TOL, abs=ABS_TOL
    )
    assert results["res_syn"]["mean_mrr"] == pytest.approx(0.2512, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_nrms_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["nrms_quickstart"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, seed=42, MIND_type="demo"),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["res_syn"]["group_auc"] == pytest.approx(
        0.5768, rel=TOL, abs=ABS_TOL
    )
    assert results["res_syn"]["mean_mrr"] == pytest.approx(0.2457, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_npa_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["npa_quickstart"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, seed=42, MIND_type="demo"),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["res_syn"]["group_auc"] == pytest.approx(
        0.5861, rel=TOL, abs=ABS_TOL
    )
    assert results["res_syn"]["mean_mrr"] == pytest.approx(0.255, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_lstur_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["lstur_quickstart"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(epochs=1, seed=40, MIND_type="demo"),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["res_syn"]["group_auc"] == pytest.approx(
        0.5977, rel=TOL, abs=ABS_TOL
    )
    assert results["res_syn"]["mean_mrr"] == pytest.approx(0.2618, rel=TOL, abs=ABS_TOL)


@pytest.mark.notebooks
@pytest.mark.smoke
@pytest.mark.gpu
def test_cornac_bivae_smoke(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["cornac_bivae_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["map"] == pytest.approx(0.146552, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.474124, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.412527, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.225064, rel=TOL, abs=ABS_TOL)
