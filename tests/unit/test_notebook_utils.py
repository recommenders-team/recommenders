# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
from reco_utils.common.notebook_utils import is_jupyter


def test_is_jupyter():
    """Only can test if the module is running on non-Databricks
    """
    # Test on the terminal
    assert is_jupyter() is False

    # Test on Jupyter notebook
    pm.execute_notebook(
        'test_notebook_utils.ipynb',
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
    )
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.dataframe
    result_is_jupyter = df.loc[df["name"] == "is_jupyter", "value"].values[0]
    assert result_is_jupyter is True

# @pytest.mark.notebooks
# def test_is_databricks():
#     TODO Currently, we cannot pytest modules on Databricks
