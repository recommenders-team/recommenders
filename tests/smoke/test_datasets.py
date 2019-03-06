# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.dataset import movielens, criteo_dac
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


@pytest.mark.smoke
def test_movielens_load_pandas_df():
    size = "100k"
    df = movielens.load_pandas_df(size=size)
    assert len(df) == 100000
    assert len(df.columns) == 4

    # Test if can handle different size of header columns
    header = ["a"]
    df = movielens.load_pandas_df(header=header)
    assert len(df.columns) == len(header)

    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = movielens.load_pandas_df(header=header)
        assert len(df.columns) == 4

    # Test title load
    df = movielens.load_pandas_df(size=size, title_col="Title")
    assert len(df.columns) == 5
    # Movie 1 is Toy Story
    title = df.loc[df[DEFAULT_ITEM_COL] == 1][:2]["Title"].values
    assert title[0] == title[1]
    assert title[0] == "Toy Story (1995)"

    # Test genres load
    df = movielens.load_pandas_df(size=size, genres_col="Genres")
    assert len(df.columns) == 5
    # Movie 1 is Toy Story
    genres = df.loc[df[DEFAULT_ITEM_COL] == 1][:2]["Genres"].values
    assert genres[0] == genres[1]
    assert genres[0] == "Animation|Children's|Comedy"

    # Test movie data load (not rating data)
    df = movielens.load_pandas_df(size=size, header=None, title_col="Title", genres_col="Genres")
    assert len(df) == 1682
    assert len(df.columns) == 3

    
@pytest.mark.smoke
@pytest.mark.spark
def test_movielens_load_spark_df(spark):
    size = "100k"

    # Check if the function load correct dataset
    df = movielens.load_spark_df(spark, size=size)
    assert df.count() == 100000
    assert len(df.columns) == 4

    # Test if can handle different size of header columns
    header = ["a"]
    df = movielens.load_spark_df(spark, header=header)
    assert len(df.columns) == len(header)

    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = movielens.load_spark_df(spark, header=header)
        assert len(df.columns) == 4

    # Test title load
    df = movielens.load_spark_df(spark, size=size, title_col="Title")
    assert len(df.columns) == 5
    # Movie 1 is Toy Story
    title = df.filter(col(DEFAULT_ITEM_COL) == 1).select("Title").limit(2).collect()
    assert title[0][0] == title[1][0]
    assert title[0][0] == "Toy Story (1995)"

    # Test genres load
    df = movielens.load_spark_df(spark, size=size, genres_col="Genres")
    assert len(df.columns) == 5
    # Movie 1 is Toy Story
    genres = df.filter(col(DEFAULT_ITEM_COL) == 1).select("Genres").limit(2).collect()
    assert genres[0][0] == genres[1][0]
    assert genres[0][0] == "Animation|Children's|Comedy"

    # Test movie data load (not rating data)
    df = movielens.load_spark_df(spark, size=size, header=None, title_col="Title", genres_col="Genres")
    assert df.count() == 1682
    assert len(df.columns) == 3

    # Test if can handle wrong size argument
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, size='10k')
    # Test if can handle wrong cache path argument
    with pytest.raises(ValueError):
        movielens.load_spark_df(spark, local_cache_path='.')

    # Test if use schema when both schema and header are provided
    header = ["1", "2"]
    schema = StructType([StructField("u", IntegerType())])
    with pytest.warns(Warning):
        df = movielens.load_spark_df(spark, header=header, schema=schema)
        assert len(df.columns) == len(schema)


@pytest.mark.smoke
@pytest.mark.spark
def test_criteo_load_spark_df(spark):
    df = load_spark_df(spark, size="sample")
    assert df.count() == 100000
    assert len(df.columns) == 40
    first_row = df.limit(1).collect()[0].asDict()
    assert first_row == criteo_first_row
