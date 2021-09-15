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
def test_template_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["template"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        parameters=dict(PM_VERSION=pm.__version__),
        kernel_name=kernel_name,
    )
    nb = sb.read_notebook(output_notebook)
    df = nb.papermill_dataframe
    assert df.shape[0] == 2
    check_version = df.loc[df["name"] == "checked_version", "value"].values[0]
    assert check_version is True


@pytest.mark.notebooks
def test_sasrec_single_node_runs(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["sasrec_quickstart"]
    pm.execute_notebook(notebook_path, output_notebook, kernel_name=kernel_name)


