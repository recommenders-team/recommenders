# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import pandas as pd
from reco_utils.dataset import criteo


@pytest.mark.smoke
def test_criteo_load_pandas_df(criteo_first_row):
    df = criteo.load_pandas_df(size="sample")
    assert df.shape[0] == 100000
    assert df.shape[1] == 40
    assert df.loc[0].equals(pd.Series(criteo_first_row))


@pytest.mark.smoke
@pytest.mark.spark
def test_criteo_load_spark_df(spark, criteo_first_row):
    df = criteo.load_spark_df(spark, size="sample")
    assert df.count() == 100000
    assert len(df.columns) == 40
    first_row = df.limit(1).collect()[0].asDict()
    assert first_row == criteo_first_row


@pytest.mark.smoke
def test_download_criteo(tmp_path):
    filepath = criteo.download_criteo(size="sample", work_directory=tmp_path)
    statinfo = os.stat(filepath)
    assert statinfo.st_size == 8787154


@pytest.mark.smoke
def test_extract_criteo(tmp_path):
    filepath = criteo.download_criteo(size="sample", work_directory=tmp_path)
    filename = criteo.extract_criteo(size="sample", compressed_file=filepath)
    statinfo = os.stat(filename)
    assert statinfo.st_size == 24328072
