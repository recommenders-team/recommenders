# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.05


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        ("1m", {"map": 0.064012679, "ndcg": 0.308012195, "precision": 0.277214771, "recall": 0.109291553}),
        ("10m", {"map": 0.101402403, "ndcg": 0.321072689, "precision": 0.275765514, "recall": 0.156483292}),
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
        assert results[key] == pytest.approx(value, rel=TOL)


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, expected_values",
    [
        ("1m", dict(rmse=0.89,
                    mae=0.70,
                    rsquared=0.36,
                    exp_var=0.36,
                    map=0.011,
                    ndcg=0.10,
                    precision=0.093,
                    recall=0.025)),
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
        assert results[key] == pytest.approx(value, rel=TOL)

