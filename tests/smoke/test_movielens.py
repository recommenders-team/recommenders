# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.dataset import movielens
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
    from reco_utils.common.spark_utils import start_or_get_spark
except ImportError:
    pass  # skip this import if we are in pure python environment


@pytest.mark.smoke
@pytest.mark.parametrize(
    "size, num_samples, num_movies, title_example, genres_example",
    [
        ("10m", 10000054, 10681, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
        ("20m", 20000263, 27278, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
    ],
)
def test_load_pandas_df(size, num_samples, num_movies, title_example, genres_example):
    """Test MovieLens dataset load into pd.DataFrame
    """
    df = movielens.load_pandas_df(size=size)
    assert len(df) == num_samples
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
    assert title[0] == title_example

    # Test genres load
    df = movielens.load_pandas_df(size=size, genres_col="Genres")
    assert len(df.columns) == 5
    # Movie 1 is Toy Story
    genres = df.loc[df[DEFAULT_ITEM_COL] == 1][:2]["Genres"].values
    assert genres[0] == genres[1]
    assert genres[0] == genres_example

    # Test movie data load (not rating data)
    df = movielens.load_pandas_df(size=size, header=None, title_col="Title", genres_col="Genres")
    assert len(df) == num_movies
    assert len(df.columns) == 3

    
@pytest.mark.smoke
@pytest.mark.spark
@pytest.mark.parametrize(
    "size, num_samples, num_movies, title_example, genres_example",
    [
        ("10m", 10000054, 10681, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
        ("20m", 20000263, 27278, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
    ],
)
def test_load_spark_df(size, num_samples, num_movies, title_example, genres_example):
    """Test MovieLens dataset load into pySpark.DataFrame
    """
    spark = start_or_get_spark("MovieLensLoaderTesting")

    # Check if the function load correct dataset
    df = movielens.load_spark_df(spark, size=size)
    assert df.count() == num_samples
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
    assert title[0][0] == title_example

    # Test genres load
    df = movielens.load_spark_df(spark, size=size, genres_col="Genres")
    assert len(df.columns) == 5
    # Movie 1 is Toy Story
    genres = df.filter(col(DEFAULT_ITEM_COL) == 1).select("Genres").limit(2).collect()
    assert genres[0][0] == genres[1][0]
    assert genres[0][0] == genres_example

    # Test movie data load (not rating data)
    df = movielens.load_spark_df(spark, size=size, header=None, title_col="Title", genres_col="Genres")
    assert df.count() == num_movies
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
