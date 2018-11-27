# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.dataset import movielens

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        StringType,
        FloatType,
        DoubleType,
    )
    from reco_utils.common.spark_utils import start_or_get_spark
except:
    pass  # skip this import if we are in pure python environment


def test_load_pandas_df():
    """Test MovieLens dataset load into pd.DataFrame
    """

    # Test if the function load correct dataset
    size_100k = movielens.load_pandas_df(size="100k")
    assert len(size_100k) == 100000
    assert len(size_100k.columns) == 4
    size_1m = movielens.load_pandas_df(size="1m")
    assert len(size_1m) == 1000209
    assert len(size_1m.columns) == 4
    size_10m = movielens.load_pandas_df(size="10m")
    assert len(size_10m) == 10000054
    assert len(size_10m.columns) == 4
    size_20m = movielens.load_pandas_df(size="20m")
    assert len(size_20m) == 20000263
    assert len(size_20m.columns) == 4

    # Test if can handle wrong size argument
    with pytest.raises(ValueError):
        movielens.load_pandas_df(size="10k")
    # Test if can handle wrong cache path argument
    with pytest.raises(ValueError):
        movielens.load_pandas_df(local_cache_path=".")

    # Test if can handle different size of header columns
    header = ["a", "b", "c"]
    with_header = movielens.load_pandas_df(header=header)
    assert len(with_header) == 100000
    assert len(with_header.columns) == len(header)

    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        with_header = movielens.load_pandas_df(header=header)
        assert len(with_header) == 100000
        assert len(with_header.columns) == 4


@pytest.mark.spark
def test_load_spark_df():
    """Test MovieLens dataset load into pySpark.DataFrame
    """
    spark = start_or_get_spark("MovieLensLoaderTesting")

    # Check if the function load correct dataset
    size_100k = movielens.load_spark_df(spark, size="100k")
    assert size_100k.count() == 100000
    assert len(size_100k.columns) == 4
    size_1m = movielens.load_spark_df(spark, size="1m")
    assert size_1m.count() == 1000209
    assert len(size_1m.columns) == 4
    size_10m = movielens.load_spark_df(spark, size="10m")
    assert size_10m.count() == 10000054
    assert len(size_10m.columns) == 4
    size_20m = movielens.load_spark_df(spark, size="20m")
    assert size_20m.count() == 20000263
    assert len(size_20m.columns) == 4

    # Test if can handle wrong size argument
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, size='10k')
    # Test if can handle wrong cache path argument
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, local_cache_path='.')

    # Test if can handle different size of header columns
    header = ["a", "b", "c"]
    with_header = movielens.load_spark_df(spark, header=header)
    assert with_header.count() == 100000
    assert len(with_header.columns) == len(header)

    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        with_header = movielens.load_spark_df(spark, header=header)
        assert with_header.count() == 100000
        assert len(with_header.columns) == 4

    # Test if can throw exception for wrong types
    schema = StructType([StructField("u", StringType())])
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, schema=schema)
    schema = StructType(
        [StructField("u", IntegerType()), StructField("i", StringType())]
    )
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, schema=schema)
    schema = StructType(
        [
            StructField("u", IntegerType()),
            StructField("i", IntegerType()),
            StructField("r", IntegerType()),
        ]
    )
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, schema=schema)

    # Test if can handle different size of schema fields
    schema = StructType(
        [
            StructField("u", IntegerType()),
            StructField("i", IntegerType()),
            StructField("r", FloatType()),
        ]
    )
    with_schema = movielens.load_spark_df(spark, schema=schema)
    assert with_schema.count() == 100000
    assert len(with_schema.columns) == len(schema)
    schema = StructType(
        [
            StructField("u", IntegerType()),
            StructField("i", IntegerType()),
            StructField("r", DoubleType()),
            StructField("a", IntegerType()),
            StructField("b", IntegerType()),
        ]
    )
    with pytest.warns(Warning):
        with_schema = movielens.load_spark_df(spark, schema=schema)
        assert with_schema.count() == 100000
        assert len(with_schema.columns) == 4

    # Test if use schema when both schema and header are provided
    schema = StructType([StructField("u", IntegerType())])
    with pytest.warns(Warning):
        with_schema = movielens.load_spark_df(spark, header=header, schema=schema)
        assert with_schema.count() == 100000
        assert len(with_schema.columns) == len(schema)
