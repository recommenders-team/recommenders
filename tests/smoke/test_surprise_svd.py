# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.05


@pytest.mark.smoke
def test_surprise_svd_smoke(notebooks):
    notebook_path = notebooks["surprise_svd_deep_dive"]
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(MOVIELENS_DATA_SIZE="100k"),
    )
    results = pm.read_notebook(OUTPUT_NOTEBOOK).dataframe.set_index("name")["value"]

    assert results["rmse"] == pytest.approx(0.96, TOL)
    assert results["mae"] == pytest.approx(0.75, TOL)
    assert results["rsquared"] == pytest.approx(0.29, TOL)
    assert results["exp_var"] == pytest.approx(0.29, TOL)
    assert results["map"] == pytest.approx(0.013, TOL)
    assert results["ndcg"] == pytest.approx(0.1, TOL)
    assert results["precision"] == pytest.approx(0.095, TOL)
    assert results["recall"] == pytest.approx(0.032, TOL)
