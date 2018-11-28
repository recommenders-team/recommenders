# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.5


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, result_list",
    [
        ("1m", [0.89, 0.70, 0.36, 0.36]),
        # ("10m", [0.56, 0.43, 0.72, 0.72]),  # works but pretty long
    ],
)
def test_surprise_svd_integration(notebooks, size, result_list):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE=size),
    )
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.dataframe
    result_rmse = df.loc[df["name"] == "rmse", "value"].values[0]
    assert result_rmse == pytest.approx(result_list[0], TOL)
    result_mae = df.loc[df["name"] == "mae", "value"].values[0]
    assert result_mae == pytest.approx(result_list[1], TOL)
    result_rsquared = df.loc[df["name"] == "rsquared", "value"].values[0]
    assert result_rsquared == pytest.approx(result_list[2], TOL)
    result_exp_var = df.loc[df["name"] == "exp_var", "value"].values[0]
    assert result_exp_var == pytest.approx(result_list[3], TOL)
