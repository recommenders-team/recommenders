# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import pandas as pd
from recommenders.datasets import criteo


@pytest.mark.integration
def test_criteo_load_pandas_df(criteo_first_row):
    df = criteo.load_pandas_df(size="full")
    assert df.shape[0] == 45840617
    assert df.shape[1] == 40
    assert df.loc[0].equals(pd.Series(criteo_first_row))
    del df
    gc.collect()


@pytest.mark.spark
@pytest.mark.integration
def test_criteo_load_spark_df(spark, criteo_first_row):
    df = criteo.load_spark_df(spark, size="full")
    assert df.count() == 45840617
    assert len(df.columns) == 40
    first_row = df.limit(1).collect()[0].asDict()
    assert first_row == criteo_first_row
    del df
    gc.collect()
