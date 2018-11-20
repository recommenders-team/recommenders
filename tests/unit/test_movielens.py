# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.dataset import movielens
try:
    from reco_utils.common.spark_utils import start_or_get_spark
except:
    pass  # skip this import if we are in pure python environment


def test_load_pandas_df():
    """Test MovieLens dataset load into pd.DataFrame
    """
    size_100k = len(movielens.load_pandas_df())
    assert size_100k == 100000
    size_1m = len(movielens.load_pandas_df(size="1m"))
    assert size_1m == 1000209
    size_10m = len(movielens.load_pandas_df(size="10m"))
    assert size_10m == 10000054
    size_20m = len(movielens.load_pandas_df(size="20m"))
    assert size_20m == 20000263

    with pytest.raises(ValueError):
        movielens.load_pandas_df(size='10k')
    with pytest.raises(ValueError):
        movielens.load_pandas_df(local_cache_path='.')


@pytest.mark.spark
def test_load_spark_df():
    """Test MovieLens dataset load into pySpark.DataFrame
    """
    spark = start_or_get_spark("MovieLensLoaderTesting")

    size_100k = movielens.load_spark_df(spark).count()
    assert size_100k == 100000
    size_1m = movielens.load_spark_df(spark, size="1m").count()
    assert size_1m == 1000209
    size_10m = movielens.load_spark_df(spark, size="10m").count()
    assert size_10m == 10000054
    size_20m = movielens.load_spark_df(spark, size="20m").count()
    assert size_20m == 20000263

    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, size='10k')
    with pytest.raises(ValueError):
        movielens.load_pandas_df(spark, local_cache_path='.')
