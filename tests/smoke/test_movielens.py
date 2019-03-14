# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import shutil
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
    "size, num_samples, num_movies, title_example, genres_example, year_example",
    [
        ("100k", 100000, 1682, "Toy Story (1995)", "Animation|Children's|Comedy", "1995"),
    ],
)
def test_load_pandas_df(size, num_samples, num_movies, title_example, genres_example, year_example):
    """Test MovieLens dataset load into pd.DataFrame
    """
    filename = "ml.zip"
    local_cache_dir = os.path.join("data", size)
    local_cache_path = os.path.join(local_cache_dir, filename)

    df = movielens.load_pandas_df(size=size, local_cache_path=local_cache_path)
    assert len(df) == num_samples
    assert len(df.columns) == 4

    # Test if can handle different size of header columns
    header = ["a"]
    df = movielens.load_pandas_df(size=size, header=header, local_cache_path=local_cache_path)
    assert len(df.columns) == len(header)

    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = movielens.load_pandas_df(size=size, header=header, local_cache_path=local_cache_path)
        assert len(df.columns) == 4

    # Test title, genres, and released year load
    df = movielens.load_pandas_df(size=size, local_cache_path=local_cache_path,
                                  title_col="Title", genres_col="Genres", year_col="Year")
    assert len(df.columns) == 7

    # Get first two records of MovieID == 1, which is Toy Story
    head = df.loc[df[DEFAULT_ITEM_COL] == 1][:2]

    title = head["Title"].values
    assert title[0] == title[1]         # Check if two records are the same movie
    assert title[0] == title_example    # Check if the record is Toy Story

    genres = head["Genres"].values
    assert genres[0] == genres[1]
    assert genres[0] == genres_example

    year = head["Year"].values
    assert year[0] == year[1]
    assert year[0] == year_example

    shutil.rmtree(local_cache_dir, ignore_errors=True)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "size, num_movies, title_example, genres_example, year_example",
    [
        ("100k", 1682, "Toy Story (1995)", "Animation|Children's|Comedy", "1995"),
    ],
)
def test_load_item_df(size, num_movies, title_example, genres_example, year_example):
    """Test movielens item data load (not rating data)
    """
    filename = "ml.zip"
    local_cache_dir = os.path.join("data", size)
    local_cache_path = os.path.join(local_cache_dir, filename)
    df = movielens.load_item_df(size, local_cache_path=local_cache_path)
    assert len(df) == num_movies

    # Test title and genres
    df = movielens.load_item_df(size, local_cache_path=local_cache_path, title_col="title", genres_col="genres")
    assert len(df) == num_movies
    assert df['title'][0] == title_example
    assert df['genres'][0] == genres_example

    # Test released year
    df = movielens.load_item_df(size, local_cache_path=local_cache_path, year_col="year")
    assert len(df) == num_movies
    assert df['year'][0] == year_example

    shutil.rmtree(local_cache_dir, ignore_errors=True)


@pytest.mark.smoke
@pytest.mark.parametrize("size", ["100k"])
def test_download_movielens(size):
    """Test movielens data download
    """
    filename = "ml.zip"
    movielens.download_movielens(size, filename)
    assert os.path.exists(filename)
    os.remove(filename)


@pytest.mark.smoke
@pytest.mark.spark
@pytest.mark.parametrize(
    "size, num_samples, num_movies, title_example, genres_example, year_example",
    [
        ("100k", 100000, 1682, "Toy Story (1995)", "Animation|Children's|Comedy", "1995"),
    ],
)
def test_load_spark_df(size, num_samples, num_movies, title_example, genres_example, year_example):
    """Test MovieLens dataset load into pySpark.DataFrame
    """
    spark = start_or_get_spark("MovieLensLoaderTesting")

    filename = "ml.zip"
    local_cache_dir = os.path.join("data", size)
    local_cache_path = os.path.join(local_cache_dir, filename)

    # Check if the function load correct dataset
    df = movielens.load_spark_df(spark, size=size, local_cache_path=local_cache_path)
    assert df.count() == num_samples
    assert len(df.columns) == 4

    # Test if can handle different size of header columns
    header = ["1", "2"]
    schema = StructType([StructField("u", IntegerType())])
    with pytest.warns(Warning):
        # Test if use schema when both schema and header are provided
        df = movielens.load_spark_df(spark, size=size, header=header, schema=schema, local_cache_path=local_cache_path)
        assert df.count() == num_samples
        assert len(df.columns) == len(schema)
    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = movielens.load_spark_df(spark, size=size, header=header, local_cache_path=local_cache_path)
        assert df.count() == num_samples
        assert len(df.columns) == 4

    # Test title load
    df = movielens.load_spark_df(spark, size=size, local_cache_path=local_cache_path,
                                 title_col="Title", genres_col="Genres", year_col="Year")
    assert df.count() == num_samples
    assert len(df.columns) == 7

    # Get first two records of MovieID == 1, which is Toy Story
    head = df.filter(col(DEFAULT_ITEM_COL) == 1).limit(2)

    title = head.select("Title").collect()
    assert title[0][0] == title[1][0]       # Check if two records are the same movie
    assert title[0][0] == title_example     # Check if the record is Toy Story

    genres = head.select("Genres").collect()
    assert genres[0][0] == genres[1][0]
    assert genres[0][0] == genres_example

    year = head.select("Year").collect()
    assert year[0][0] == year[1][0]
    assert year[0][0] == year_example

    shutil.rmtree(local_cache_dir, ignore_errors=True)
