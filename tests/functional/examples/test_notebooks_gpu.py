# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pytest

from recommenders.utils.gpu_utils import get_number_gpus
from recommenders.utils.notebook_utils import execute_notebook, read_notebook


TOL = 0.1
ABS_TOL = 0.05


@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "size, epochs, expected_values, seed",
    [
        (
            "1m",
            10,
            {
                "map": 0.0255283,
                "ndcg": 0.15656,
                "precision": 0.145646,
                "recall": 0.0557367,
            },
            42,
        ),
        # ("10m", 5, {"map": 0.024821, "ndcg": 0.153396, "precision": 0.143046, "recall": 0.056590})# takes too long
    ],
)
def test_ncf_functional(
    notebooks, output_notebook, kernel_name, size, epochs, expected_values, seed
):
    notebook_path = notebooks["ncf"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10, MOVIELENS_DATA_SIZE=size, EPOCHS=epochs, BATCH_SIZE=512, SEED=seed
        ),
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "size, epochs, batch_size, expected_values, seed",
    [
        (
            "100k",
            10,
            512,
            {
                "map": 0.0435856,
                "ndcg": 0.37586,
                "precision": 0.169353,
                "recall": 0.0923963,
                "map2": 0.0510391,
                "ndcg2": 0.202186,
                "precision2": 0.179533,
                "recall2": 0.106434,
            },
            42,
        )
    ],
)
def test_ncf_deep_dive_functional(
    notebooks,
    output_notebook,
    kernel_name,
    size,
    epochs,
    batch_size,
    expected_values,
    seed,
):
    notebook_path = notebooks["ncf_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10,
            MOVIELENS_DATA_SIZE=size,
            EPOCHS=epochs,
            BATCH_SIZE=batch_size,
            SEED=seed,
        ),
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
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
def test_embdotbias_functional(
    notebooks, output_notebook, kernel_name, size, epochs, expected_values
):
    notebook_path = notebooks["embdotbias"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE=size, EPOCHS=epochs),
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "epochs, expected_values, seed",
    [
        (
            5,
            {"auc": 0.742, "logloss": 0.4964},
            42,
        )
    ],
)
def test_xdeepfm_functional(
    notebooks,
    output_notebook,
    kernel_name,
    epochs,
    expected_values,
    seed,
):
    notebook_path = notebooks["xdeepfm_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            EPOCHS=epochs,
            BATCH_SIZE=1024,
            RANDOM_SEED=seed,
        ),
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "size, steps, batch_size, expected_values, seed",
    [
        (
            "100k",
            10000,
            32,
            {
                "rmse": 0.924958,
                "mae": 0.741425,
                "rsquared": 0.262963,
                "exp_var": 0.268413,
                "ndcg_at_k": 0.118114,
                "map": 0.0139213,
                "precision_at_k": 0.107087,
                "recall_at_k": 0.0328638,
            },
            42,
        )
    ],
)
def test_wide_deep_functional(
    notebooks,
    output_notebook,
    kernel_name,
    size,
    steps,
    batch_size,
    expected_values,
    seed,
    tmp,
):
    notebook_path = notebooks["wide_deep"]

    params = {
        "MOVIELENS_DATA_SIZE": size,
        "STEPS": steps,
        "BATCH_SIZE": batch_size,
        "EVALUATE_WHILE_TRAINING": False,
        "MODEL_DIR": tmp,
        "EXPORT_DIR_BASE": tmp,
        "RATING_METRICS": ["rmse", "mae", "rsquared", "exp_var"],
        "RANKING_METRICS": ["ndcg_at_k", "map", "precision_at_k", "recall_at_k"],
        "RANDOM_SEED": seed,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "yaml_file, data_path, epochs, batch_size, expected_values, seed",
    [
        (
            "recommenders/models/deeprec/config/sli_rec.yaml",
            os.path.join("tests", "resources", "deeprec", "slirec"),
            10,
            400,
            {
                "auc": 0.7183
            },  # Don't do logloss check as SLi-Rec uses ranking loss, not a point-wise loss
            42,
        )
    ],
)
def test_slirec_quickstart_functional(
    notebooks,
    output_notebook,
    kernel_name,
    yaml_file,
    data_path,
    epochs,
    batch_size,
    expected_values,
    seed,
):
    notebook_path = notebooks["slirec_quickstart"]

    params = {
        "yaml_file": yaml_file,
        "data_path": data_path,
        "EPOCHS": epochs,
        "BATCH_SIZE": batch_size,
        "RANDOM_SEED": seed,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    assert results["auc"] == pytest.approx(expected_values["auc"], rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "epochs, batch_size, seed, MIND_type, expected_values",
    [
        (
            5,
            64,
            42,
            "demo",
            {
                "group_auc": 0.6217,
                "mean_mrr": 0.2783,
                "ndcg@5": 0.3024,
                "ndcg@10": 0.3719,
            },
        )
    ],
)
def test_nrms_quickstart_functional(
    notebooks,
    output_notebook,
    kernel_name,
    epochs,
    batch_size,
    seed,
    MIND_type,
    expected_values,
):
    notebook_path = notebooks["nrms_quickstart"]

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "MIND_type": MIND_type,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        expected_values["group_auc"], rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(
        expected_values["mean_mrr"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@5"] == pytest.approx(
        expected_values["ndcg@5"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@10"] == pytest.approx(
        expected_values["ndcg@10"], rel=TOL, abs=ABS_TOL
    )


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "epochs, batch_size, seed, MIND_type, expected_values",
    [
        (
            5,
            64,
            42,
            "demo",
            {
                "group_auc": 0.6436,
                "mean_mrr": 0.2990,
                "ndcg@5": 0.3297,
                "ndcg@10": 0.3933,
            },
        )
    ],
)
def test_naml_quickstart_functional(
    notebooks,
    output_notebook,
    kernel_name,
    batch_size,
    epochs,
    seed,
    MIND_type,
    expected_values,
):
    notebook_path = notebooks["naml_quickstart"]

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "MIND_type": MIND_type,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        expected_values["group_auc"], rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(
        expected_values["mean_mrr"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@5"] == pytest.approx(
        expected_values["ndcg@5"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@10"] == pytest.approx(
        expected_values["ndcg@10"], rel=TOL, abs=ABS_TOL
    )


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "epochs, batch_size, seed, MIND_type, expected_values",
    [
        (
            5,
            64,
            42,
            "demo",
            {
                "group_auc": 0.6444,
                "mean_mrr": 0.2983,
                "ndcg@5": 0.3287,
                "ndcg@10": 0.3938,
            },
        )
    ],
)
def test_lstur_quickstart_functional(
    notebooks,
    output_notebook,
    kernel_name,
    epochs,
    batch_size,
    seed,
    MIND_type,
    expected_values,
):
    notebook_path = notebooks["lstur_quickstart"]

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "MIND_type": MIND_type,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        expected_values["group_auc"], rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(
        expected_values["mean_mrr"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@5"] == pytest.approx(
        expected_values["ndcg@5"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@10"] == pytest.approx(
        expected_values["ndcg@10"], rel=TOL, abs=ABS_TOL
    )


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "epochs, batch_size, seed, MIND_type, expected_values",
    [
        (
            5,
            64,
            42,
            "demo",
            {
                "group_auc": 0.6035,
                "mean_mrr": 0.2765,
                "ndcg@5": 0.2977,
                "ndcg@10": 0.3637,
            },
        )
    ],
)
def test_npa_quickstart_functional(
    notebooks,
    output_notebook,
    kernel_name,
    epochs,
    batch_size,
    seed,
    MIND_type,
    expected_values,
):
    notebook_path = notebooks["npa_quickstart"]

    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "MIND_type": MIND_type,
    }
    execute_notebook(
        notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
    )
    results = read_notebook(output_notebook)

    assert results["group_auc"] == pytest.approx(
        expected_values["group_auc"], rel=TOL, abs=ABS_TOL
    )
    assert results["mean_mrr"] == pytest.approx(
        expected_values["mean_mrr"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@5"] == pytest.approx(
        expected_values["ndcg@5"], rel=TOL, abs=ABS_TOL
    )
    assert results["ndcg@10"] == pytest.approx(
        expected_values["ndcg@10"], rel=TOL, abs=ABS_TOL
    )


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "yaml_file, data_path, size, epochs, batch_size, expected_values, seed",
    [
        (
            "recommenders/models/deeprec/config/lightgcn.yaml",
            os.path.join("tests", "resources", "deeprec", "lightgcn"),
            "100k",
            5,
            1024,
            {
                "map": 0.094794,
                "ndcg": 0.354145,
                "precision": 0.308165,
                "recall": 0.163034,
            },
            42,
        )
    ],
)
def test_lightgcn_deep_dive_functional(
    notebooks,
    output_notebook,
    kernel_name,
    yaml_file,
    data_path,
    size,
    epochs,
    batch_size,
    expected_values,
    seed,
):
    notebook_path = notebooks["lightgcn_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            TOP_K=10,
            MOVIELENS_DATA_SIZE=size,
            EPOCHS=epochs,
            BATCH_SIZE=batch_size,
            SEED=seed,
            yaml_file=yaml_file,
            user_file=os.path.join(data_path, r"user_embeddings"),
            item_file=os.path.join(data_path, r"item_embeddings"),
        ),
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
def test_dkn_quickstart_functional(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["dkn_quickstart"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(EPOCHS=5, BATCH_SIZE=200),
    )
    results = read_notebook(output_notebook)

    assert results["auc"] == pytest.approx(0.5651, rel=TOL, abs=ABS_TOL)
    assert results["mean_mrr"] == pytest.approx(0.1639, rel=TOL, abs=ABS_TOL)
    assert results["ndcg@5"] == pytest.approx(0.1735, rel=TOL, abs=ABS_TOL)
    assert results["ndcg@10"] == pytest.approx(0.2301, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "size, expected_values",
    [
        ("1m", dict(map=0.081794, ndcg=0.400983, precision=0.367997, recall=0.138352)),
        # 10m works but takes too long
    ],
)
def test_cornac_bivae_functional(
    notebooks, output_notebook, kernel_name, size, expected_values
):
    notebook_path = notebooks["cornac_bivae_deep_dive"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(MOVIELENS_DATA_SIZE=size),
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "data_dir, num_epochs, batch_size, model_name, expected_values",
    [
        (
            os.path.join("tests", "recsys_data", "RecSys", "SASRec-tf2", "data"),
            1,
            128,
            "sasrec",
            {"ndcg@10": 0.2297, "Hit@10": 0.3789},
        ),
        (
            os.path.join("tests", "recsys_data", "RecSys", "SASRec-tf2", "data"),
            1,
            128,
            "ssept",
            {"ndcg@10": 0.2245, "Hit@10": 0.3743},
        ),
    ],
)
def test_sasrec_quickstart_functional(
    notebooks,
    output_notebook,
    kernel_name,
    data_dir,
    num_epochs,
    batch_size,
    model_name,
    expected_values,
):
    notebook_path = notebooks["sasrec_quickstart"]
    params = {
        "data_dir": data_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "model_name": model_name,
    }
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=params,
    )
    results = read_notebook(output_notebook)

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)


@pytest.mark.gpu
@pytest.mark.notebooks
@pytest.mark.parametrize(
    "size, algos, expected_values_ndcg",
    [
        (
            ["100k"],
            ["ncf", "embdotbias", "bivae", "lightgcn"],
            [0.382793, 0.147583, 0.471722, 0.412664],
        ),
    ],
)
def test_benchmark_movielens_gpu(
    notebooks, output_notebook, kernel_name, size, algos, expected_values_ndcg
):
    notebook_path = notebooks["benchmark_movielens"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(data_sizes=size, algorithms=algos),
    )
    results = read_notebook(output_notebook)

    assert len(results) == 4
    for i, value in enumerate(algos):
        assert results[value] == pytest.approx(
            expected_values_ndcg[i], rel=TOL, abs=ABS_TOL
        )
