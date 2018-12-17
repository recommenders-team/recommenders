# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import papermill as pm
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
from reco_utils.common.spark_utils import start_or_get_spark


TOL = 0.05


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
    nb = pm.read_notebook(OUTPUT_NOTEBOOK)
    results = nb.dataframe.set_index("name")["value"]
    start_or_get_spark("ALS PySpark").stop()

    assert results["map"] == pytest.approx(0.00481, rel=TOL)
    assert results["ndcg"] == pytest.approx(0.04289, rel=TOL)
    assert results["precision"] == pytest.approx(0.047558, rel=TOL)
    assert results["recall"] == pytest.approx(0.018512, rel=TOL)
    assert results["rmse"] == pytest.approx(0.950697, rel=TOL)
    assert results["mae"] == pytest.approx(0.7424, rel=TOL)
    assert results["exp_var"] == pytest.approx(0.285606, rel=TOL)
    assert results["rsquared"] == pytest.approx(0.280812, rel=TOL)
