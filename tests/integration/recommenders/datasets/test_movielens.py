# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from recommenders.datasets.movielens import (
    load_pandas_df,
    load_spark_df,
    load_item_df,
    download_movielens,
    extract_movielens,
)
import gc

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
    )
    from pyspark.sql.functions import col
except ImportError:
    pass  # skip this import if we are in pure python environment


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, num_samples, num_movies, movie_example, title_example, genres_example, year_example",
    [
        (
            "1m",
            1000209,
            3883,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        ),
        (
            "10m",
            10000054,
            10681,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
        (
            "20m",
            20000263,
            27278,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
    ],
)
def test_load_pandas_df(
    size,
    num_samples,
    num_movies,
    movie_example,
    title_example,
    genres_example,
    year_example,
    tmp,
):
    """Test MovieLens dataset load as pd.DataFrame"""
    # Test if correct data are loaded
    header = ["a", "b", "c"]
    df = load_pandas_df(size=size, local_cache_path=tmp, header=header)
    assert len(df) == num_samples
    assert len(df.columns) == len(header)
    # Test if raw-zip file, rating file, and item file are cached
    assert len(os.listdir(tmp)) == 3

    # Test title, genres, and released year load
    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = load_pandas_df(
            size=size,
            header=header,
            local_cache_path=tmp,
            title_col="Title",
            genres_col="Genres",
            year_col="Year",
        )
        assert len(df) == num_samples
        assert (
            len(df.columns) == 7
        )  # 4 header columns (user, item, rating, timestamp) and 3 feature columns
        assert "e" not in df.columns  # only the first 4 header columns are used
        # Get two records of the same items and check if the item-features are the same.
        head = df.loc[df["b"] == movie_example][:2]
        title = head["Title"].values
        assert title[0] == title[1]
        assert title[0] == title_example
        genres = head["Genres"].values
        assert genres[0] == genres[1]
        assert genres[0] == genres_example
        year = head["Year"].values
        assert year[0] == year[1]
        assert year[0] == year_example

    # Test default arguments
    df = load_pandas_df(size)
    assert len(df) == num_samples
    # user, item, rating and timestamp
    assert len(df.columns) == 4
    del df
    gc.collect()


@pytest.mark.integration
@pytest.mark.parametrize(
    "size, num_movies, movie_example, title_example, genres_example, year_example",
    [
        ("1m", 3883, 1, "Toy Story (1995)", "Animation|Children's|Comedy", "1995"),
        (
            "10m",
            10681,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
        (
            "20m",
            27278,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
    ],
)
def test_load_item_df(
    size,
    num_movies,
    movie_example,
    title_example,
    genres_example,
    year_example,
    tmp,
):
    """Test movielens item data load (not rating data)"""
    df = load_item_df(size, local_cache_path=tmp, title_col="title")
    assert len(df) == num_movies
    # movie_col and title_col should be loaded
    assert len(df.columns) == 2
    assert df["title"][0] == title_example

    # Test title and genres
    df = load_item_df(
        size,
        local_cache_path=tmp,
        movie_col="item",
        genres_col="genres",
        year_col="year",
    )
    assert len(df) == num_movies
    # movile_col, genres_col and year_col
    assert len(df.columns) == 3

    assert df["item"][0] == movie_example
    assert df["genres"][0] == genres_example
    assert df["year"][0] == year_example
    del df
    gc.collect()


@pytest.mark.integration
@pytest.mark.spark
@pytest.mark.parametrize(
    "size, num_samples, num_movies, movie_example, title_example, genres_example, year_example",
    [
        (
            "1m",
            1000209,
            3883,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        ),
        (
            "10m",
            10000054,
            10681,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
        (
            "20m",
            20000263,
            27278,
            1,
            "Toy Story (1995)",
            "Adventure|Animation|Children|Comedy|Fantasy",
            "1995",
        ),
    ],
)
def test_load_spark_df(
    size,
    num_samples,
    num_movies,
    movie_example,
    title_example,
    genres_example,
    year_example,
    tmp,
    spark,
):
    """Test MovieLens dataset load into pySpark.DataFrame"""

    # Test if correct data are loaded
    header = ["1", "2", "3"]
    schema = StructType(
        [
            StructField("u", IntegerType()),
            StructField("m", IntegerType()),
        ]
    )
    with pytest.warns(Warning):
        df = load_spark_df(
            spark, size=size, local_cache_path=tmp, header=header, schema=schema
        )
        assert df.count() == num_samples
        # Test if schema is used when both schema and header are provided
        assert len(df.columns) == len(schema)
        # Test if raw-zip file, rating file, and item file are cached
        assert len(os.listdir(tmp)) == 3

    # Test title, genres, and released year load
    header = ["a", "b", "c", "d", "e"]
    with pytest.warns(Warning):
        df = load_spark_df(
            spark,
            size=size,
            local_cache_path=tmp,
            header=header,
            title_col="Title",
            genres_col="Genres",
            year_col="Year",
        )
        assert df.count() == num_samples
        assert (
            len(df.columns) == 7
        )  # 4 header columns (user, item, rating, timestamp) and 3 feature columns
        assert "e" not in df.columns  # only the first 4 header columns are used
        # Get two records of the same items and check if the item-features are the same.
        head = df.filter(col("b") == movie_example).limit(2)
        title = head.select("Title").collect()
        assert title[0][0] == title[1][0]
        assert title[0][0] == title_example
        genres = head.select("Genres").collect()
        assert genres[0][0] == genres[1][0]
        assert genres[0][0] == genres_example
        year = head.select("Year").collect()
        assert year[0][0] == year[1][0]
        assert year[0][0] == year_example

    # Test default arguments
    df = load_spark_df(spark, size)
    assert df.count() == num_samples
    # user, item, rating and timestamp
    assert len(df.columns) == 4
    del df
    gc.collect()


@pytest.mark.integration
@pytest.mark.parametrize("size", ["1m", "10m", "20m"])
def test_download_and_extract_movielens(size, tmp):
    """Test movielens data download and extract"""
    zip_path = os.path.join(tmp, "ml.zip")
    download_movielens(size, dest_path=zip_path)
    assert len(os.listdir(tmp)) == 1
    assert os.path.exists(zip_path)

    rating_path = os.path.join(tmp, "rating.dat")
    item_path = os.path.join(tmp, "item.dat")
    extract_movielens(
        size, rating_path=rating_path, item_path=item_path, zip_path=zip_path
    )
    # Test if raw-zip file, rating file, and item file are cached
    assert len(os.listdir(tmp)) == 3
    assert os.path.exists(rating_path)
    assert os.path.exists(item_path)
