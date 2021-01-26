# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import pytest
import papermill as pm
import scrapbook as sb

from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


TOL = 0.05
ABS_TOL = 0.05


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

    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["map"] == pytest.approx(0.0052, rel=TOL, abs=ABS_TOL)
    assert results["ndcg"] == pytest.approx(0.0463, rel=TOL, abs=ABS_TOL)
    assert results["precision"] == pytest.approx(0.0487, rel=TOL, abs=ABS_TOL)
    assert results["recall"] == pytest.approx(0.0177, rel=TOL, abs=ABS_TOL)
    assert results["rmse"] == pytest.approx(0.9636, rel=TOL, abs=ABS_TOL)
    assert results["mae"] == pytest.approx(0.7508, rel=TOL, abs=ABS_TOL)
    assert results["exp_var"] == pytest.approx(0.2672, rel=TOL, abs=ABS_TOL)
    assert results["rsquared"] == pytest.approx(0.2611, rel=TOL, abs=ABS_TOL)


@pytest.mark.smoke
@pytest.mark.spark
@pytest.mark.skipif(sys.platform == "win32", reason="Not implemented on Windows")
def test_mmlspark_lightgbm_criteo_smoke(notebooks):
    notebook_path = notebooks["mmlspark_lightgbm_criteo"]
    pm.execute_notebook(
        notebook_path,
        OUTPUT_NOTEBOOK,
        kernel_name=KERNEL_NAME,
        parameters=dict(DATA_SIZE="sample", NUM_ITERATIONS=50, EARLY_STOPPING_ROUND=10),
    )

    results = sb.read_notebook(OUTPUT_NOTEBOOK).scraps.dataframe.set_index("name")[
        "data"
    ]
    assert results["auc"] == pytest.approx(0.68895, rel=TOL, abs=ABS_TOL)
