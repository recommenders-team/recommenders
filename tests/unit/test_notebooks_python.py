import os
import pytest
import pandas as pd
import papermill as pm
from tests.unit.notebooks_common import (
    path_notebooks,
    conda_environment_name,
    OUTPUT_NOTEBOOK,
)


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()
    paths = {
        "template": os.path.join(folder_notebooks, "template.ipynb"),
        "sar_single_node": os.path.join(
            folder_notebooks, "00_quick_start", "sar_python_cpu_movielens.ipynb"
        ),
    }
    return paths


@pytest.mark.notebooks
def test_template_runs(notebooks):
    notebook_path = notebooks["template"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        parameters=dict(pm_version=pm.__version__),
        kernel_name=conda_environment_name(),
    )
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    df = nb.dataframe
    assert df.shape[0] == 2
    check_version = df.loc[df["name"] == "checked_version", "value"].values[0]
    assert check_version is True


@pytest.mark.notebooks
def test_sar_single_node_runs(notebooks):
    notebook_path = notebooks["sar_single_node"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK)
