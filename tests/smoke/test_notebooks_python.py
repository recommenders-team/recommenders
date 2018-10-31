# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import pandas as pd
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.smoke
def test_sar_single_node_smoke(notebooks):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(TOP_K=10, MOVIELENS_DATA_SIZE="100k"),
    )
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.dataframe
    result_map = df.loc[df["name"] == "map", "value"].values[0]
    assert result_map > 0.10
    result_ndcg = df.loc[df["name"] == "ndcg", "value"].values[0]
    assert result_ndcg > 0.37
    result_precision = df.loc[df["name"] == "precision", "value"].values[0]
    assert result_precision > 0.32
    result_recall = df.loc[df["name"] == "recall", "value"].values[0]
    assert result_recall > 0.17
