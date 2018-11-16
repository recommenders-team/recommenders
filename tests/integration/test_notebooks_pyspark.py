# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.5


@pytest.mark.spark
@pytest.mark.integration
@pytest.mark.parametrize(
    "size, result_list",
    [
        ("1m", [0.02, 0.15, 0.14, 0.07, 0.95, 0.73, 0.28, 0.28]),
        ("10m", [0.03, 0.15, 0.14, 0.09, 0.85, 0.65, 0.36, 0.36]),
        ("20m", [0.03, 0.15, 0.14, 0.08, 0.84, 0.63, 0.37, 0.37]),
    ],
)
def test_als_pyspark_integration(notebooks, size, result_list):
    notebook_path = notebooks["als_pyspark"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE=size),
    )
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.dataframe
    result_map = df.loc[df["name"] == "map", "value"].values[0]
    assert result_map == pytest.approx(result_list[0], TOL)
    result_ndcg = df.loc[df["name"] == "ndcg", "value"].values[0]
    assert result_ndcg == pytest.approx(result_list[1], TOL)
    result_precision = df.loc[df["name"] == "precision", "value"].values[0]
    assert result_precision == pytest.approx(result_list[2], TOL)
    result_recall = df.loc[df["name"] == "recall", "value"].values[0]
    assert result_recall == pytest.approx(result_list[3], TOL)

    result_rmse = df.loc[df["name"] == "rmse", "value"].values[0]
    assert result_rmse == pytest.approx(result_list[4], TOL)
    result_mae = df.loc[df["name"] == "mae", "value"].values[0]
    assert result_mae == pytest.approx(result_list[5], TOL)
    result_exp_var = df.loc[df["name"] == "exp_var", "value"].values[0]
    assert result_exp_var == pytest.approx(result_list[6], TOL)
    result_rsquared = df.loc[df["name"] == "rsquared", "value"].values[0]
    assert result_rsquared == pytest.approx(result_list[7], TOL)

