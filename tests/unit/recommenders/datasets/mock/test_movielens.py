from recommenders.datasets.mock.movielens import MockMovielens100kSchema
from recommenders.datasets.movielens import DEFAULT_HEADER
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)

import pytest
import pandas
import pyspark.sql
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, LongType, StructField, StructType


@pytest.fixture(scope="module")
def default_schema():
    return StructType([
        StructField(DEFAULT_USER_COL, IntegerType()),
        StructField(DEFAULT_ITEM_COL, IntegerType()),
        StructField(DEFAULT_RATING_COL, FloatType()),
        StructField(DEFAULT_TIMESTAMP_COL, LongType()),
    ])


@pytest.fixture(scope="module")
def custom_schema():
    return StructType([
        StructField("userID", IntegerType()),
        StructField("itemID", IntegerType()),
        StructField("rating", FloatType()),
    ])


@pytest.mark.parametrize("size", [10, 100])
def test_mock_movielens_schema__has_default_col_names(size):
    df = MockMovielens100kSchema.example(size=size)
    for col_name in DEFAULT_HEADER:
        assert col_name in df.columns


@pytest.mark.parametrize("seed", [-1])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [0, 3, 10])
def test_mock_movielens_schema__get_df__return_success(size, seed):
    df = MockMovielens100kSchema.get_df(size, seed=seed)
    assert type(df) == pandas.DataFrame
    assert len(df) == size


@pytest.mark.parametrize("seed", [0, 101])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [3, 10])
def test_mock_movielens_schema__get_spark_df__return_success(spark: SparkSession, size, seed):
    df = MockMovielens100kSchema.get_spark_df(spark, size, seed=seed)
    assert type(df) == pyspark.sql.DataFrame
    assert df.count() == size


@pytest.mark.parametrize("schema", [
    None,
    pytest.lazy_fixture('default_schema'),
    pytest.lazy_fixture('custom_schema')
])
def test_mock_movielens_schema__get_spark_df__with_custom_schema_return_success(spark: SparkSession, schema):
    df = MockMovielens100kSchema.get_spark_df(spark, schema=schema)
    assert type(df) == pyspark.sql.DataFrame
    assert df.count() >= 0


def test_mock_movielens_schema__get_spark_df__fail_on_empty_rows(spark: SparkSession):
    with pytest.raises(ValueError, match="can not infer schema from empty dataset.*"):
        MockMovielens100kSchema.get_spark_df(spark, 0)
