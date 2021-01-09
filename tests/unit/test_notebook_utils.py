# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
import scrapbook as sb
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
from reco_utils.common.notebook_utils import is_jupyter, is_databricks


@pytest.mark.notebooks
def test_is_jupyter():
    # Test on the terminal
    assert is_jupyter() is False
    assert is_databricks() is False

    # Test on Jupyter notebook
    path = os.path.join("tests", "unit", "test_notebook_utils.ipynb")
    pm.execute_notebook(
        path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME,
    )
    nb = sb.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.papermill_dataframe
    result_is_jupyter = df.loc[df["name"] == "is_jupyter", "value"].values[0]
    assert result_is_jupyter
    assert result_is_jupyter is True
    result_is_databricks = df.loc[df["name"] == "is_databricks", "value"].values[0]
    assert result_is_databricks is False


# @pytest.mark.notebooks
# def test_is_databricks():
#     TODO Currently, we cannot pytest modules on Databricks
