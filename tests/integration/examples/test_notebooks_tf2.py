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
    "yaml_file, data_path, epochs, batch_size, expected_values, seed",
    [
        (
            "recommenders/models/deeprec/config/sli_rec.yaml",
            os.path.join("tests", "resources", "deeprec", "slirec"),
            5,
            128,
            {"ndcg@10": 0.3014, "Hit@10": 0.4875},
            42,
        )
    ],
)
#     "size, epochs, expected_values, seed",
#     [
#         (
#             "1m",
#             5,
#             {
#                 "ndcg@10": 0.3014,
#                 "Hit@10": 0.4875,
#             },
#             42,
#         ),
#         # ("10m", 5, {"map": 0.024821, "ndcg": 0.153396, "precision": 0.143046, "recall": 0.056590})# takes too long
#     ],
# )
@pytest.mark.skipif(tf.__versoin__ > "2.0", reason="We are currently on TF 1.5")
def test_sasrec_quickstart_integration(
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
    notebook_path = notebooks["sasrec_quickstart"]
    params = {
        "yaml_file": yaml_file,
        "data_path": data_path,
        "num_epochs": epochs,
        "batch_size": batch_size,
        "RANDOM_SEED": seed,
    }

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


# @pytest.mark.gpu
# @pytest.mark.integration
# @pytest.mark.parametrize(
#     "yaml_file, data_path, epochs, batch_size, expected_values, seed",
#     [
#         (
#             "recommenders/models/deeprec/config/sli_rec.yaml",
#             os.path.join("tests", "resources", "deeprec", "slirec"),
#             5,
#             128,
#             {"res_syn": {"ndcg@10": 0.3014, "Hit@10": 0.4875}},
#             42,
#         )
#     ],
# )
# def test_sasrec_quickstart_integration(
#     notebooks,
#     output_notebook,
#     kernel_name,
#     yaml_file,
#     data_path,
#     epochs,
#     batch_size,
#     expected_values,
#     seed,
# ):
#     notebook_path = notebooks["sasrec_quickstart"]

#     params = {
#         "yaml_file": yaml_file,
#         "data_path": data_path,
#         "num_epochs": epochs,
#         "batch_size": batch_size,
#         "RANDOM_SEED": seed,
#     }
#     pm.execute_notebook(
#         notebook_path, output_notebook, kernel_name=kernel_name, parameters=params
#     )
#     results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
#         "data"
#     ]

#     for key, value in expected_values.items():
#         assert results[key]["auc"] == pytest.approx(value["auc"], rel=TOL, abs=ABS_TOL)

#         ## disable logloss check, because so far SLi-Rec uses ranking loss, not a point-wise loss
#         # assert results[key]["logloss"] == pytest.approx(
#         #     value["logloss"], rel=TOL, abs=ABS_TOL
#         # )
