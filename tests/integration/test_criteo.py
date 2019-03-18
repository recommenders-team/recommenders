# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import pandas as pd
from reco_utils.dataset import criteo
from reco_utils.common.constants import DEFAULT_ITEM_COL

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        StringType,
        FloatType,
        DoubleType,
    )
    from pyspark.sql.functions import col
except ImportError:
    pass  # skip this import if we are in pure python environment


@pytest.mark.integration
def test_criteo_load_pandas_df(criteo_first_row):
    df = criteo.load_pandas_df(size="full")
    assert df.shape[0] == 45840617
    assert df.shape[1] == 40
    assert df.loc[0].equals(pd.Series(criteo_first_row))


@pytest.mark.spark
@pytest.mark.integration
def test_criteo_load_spark_df(spark, criteo_first_row):
    df = criteo.load_spark_df(spark, size="full")
    assert df.count() == 45840617
    assert len(df.columns) == 40
    first_row = df.limit(1).collect()[0].asDict()
    assert first_row == criteo_first_row

