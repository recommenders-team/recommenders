# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest

try:
    import papermill as pm
    import scrapbook as sb
except ImportError:
    pass  # disable error while collecting tests for non-notebook environments


TOL = 0.05
ABS_TOL = 0.05


@pytest.mark.notebooks
@pytest.mark.gpu
def test_sasrec_single_node_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sasrec_quickstart"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)
