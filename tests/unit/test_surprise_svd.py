# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import pandas as pd
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.notebooks
def test_sar_single_node_runs(notebooks):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)

