# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import papermill as pm
import pytest
from reco_utils.common.gpu_utils import get_number_gpus
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.05


@pytest.mark.integration
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.integration
@pytest.mark.gpu
@pytest.mark.parametrize(
    "size, expected_values",
    [
        ("1m", {"map": 0.064012679, "ndcg": 0.308012195, "precision": 0.277214771, "recall": 0.109291553}),
        ("10m", {"map": 0.101402403, "ndcg": 0.321072689, "precision": 0.275765514, "recall": 0.156483292}),
    ],
)
def test_ncf_integration(notebooks, size, expected_values):
    notebook_path = notebooks["ncf"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE=size),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    for key, value in expected_values.items():
        assert results[key] == pytest.approx(value, rel=TOL)
