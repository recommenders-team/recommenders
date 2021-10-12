# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import tensorflow as tf

try:
    import papermill as pm
    import scrapbook as sb
except ImportError:
    pass  # disable error while collecting tests for non-notebook environments


# from recommenders.utils.gpu_utils import get_number_gpus


TOL = 0.5
ABS_TOL = 0.05


# @pytest.mark.gpu
# @pytest.mark.integration
# def test_gpu_vm():
#     assert get_number_gpus() >= 1


@pytest.mark.integration
@pytest.mark.parametrize(
    "data_dir, num_epochs, batch_size, expected_values, seed",
    [
        (
            "/recsys_data/RecSys/SASRec-tf2/data/",
            1,
            128,
            {"ndcg@10": 0.2626, "Hit@10": 0.4244},
            42,
        )
    ],
)
# @pytest.mark.skipif(tf.__versoin__ > "2.0", reason="We are currently on TF 1.5")
def test_sasrec_quickstart_integration(
    notebooks,
    output_notebook,
    kernel_name,
    data_dir,
    num_epochs,
    batch_size,
    expected_values,
    seed,
):
    notebook_path = notebooks["sasrec_quickstart"]
    params = {
        "data_dir": data_dir,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "seed": seed,
    }

    print("Executing notebook ... ")
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=params,
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL, abs=ABS_TOL)
