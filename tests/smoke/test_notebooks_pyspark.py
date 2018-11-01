# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.5


@pytest.mark.smoke
@pytest.mark.spark
def test_als_pyspark_smoke(notebooks):
    notebook_path = notebooks["als_pyspark"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.dataframe
    result_map = df.loc[df["name"] == "map", "value"].values[0]
    assert result_map == pytest.approx(0.02, TOL)
    result_ndcg = df.loc[df["name"] == "ndcg", "value"].values[0]
    assert result_ndcg == pytest.approx(0.10, TOL)
    result_precision = df.loc[df["name"] == "precision", "value"].values[0]
    assert result_precision == pytest.approx(0.10, TOL)
    result_recall = df.loc[df["name"] == "recall", "value"].values[0]
    assert result_recall == pytest.approx(0.07, TOL)

