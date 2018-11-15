# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
from reco_utils.common.notebook_utils import is_jupyter, is_databricks


@pytest.mark.notebooks
def test_is_jupyter():
    """Only can test if the module is running on non-Databricks
    """
    assert is_jupyter() is False
    # Test the function on Jupyter notebook
    pm.execute_notebook(
        'test_notebook_utils.ipynb',
        OUTPUT_NOTEBOOK,
        parameters=dict(FUNCTION_NAME='is_jupyter'),
        kernel_name=KERNEL_NAME,
    )


@pytest.mark.notebooks
def test_is_databricks():
    """Only can test if the module is running on non-Databricks
    """
    assert is_databricks() is False
    # Test the function on Jupyter notebook
    pm.execute_notebook(
        'test_notebook_utils.ipynb',
        OUTPUT_NOTEBOOK,
        parameters=dict(FUNCTION_NAME='is_databricks'),
        kernel_name=KERNEL_NAME,
    )
