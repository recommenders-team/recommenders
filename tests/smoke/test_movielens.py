# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from tempfile import TemporaryDirectory
from reco_utils.dataset.movielens import (
    load_pandas_df,
    load_spark_df,
    load_item_df,
    download_movielens,
    extract_movielens,
)

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
    "size, num_samples, num_movies, movie_example, title_example, genres_example, year_example",
    [
        (
            "100k",
            100000,
            1682,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        )
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
):
    """Test MovieLens dataset load into pd.DataFrame
    """
    # Test if correct data are loaded and local_cache_path works
    with TemporaryDirectory() as tmp_dir:
        # Test if can handle different size of header columns
        header = ["a"]
        df = load_pandas_df(size=size, local_cache_path=tmp_dir, header=header)
        assert len(df) == num_samples
        assert len(df.columns) == max(
            len(header), 2
        )  # Should load at least 2 columns, user and item

        # Test title, genres, and released year load
        header = ["a", "b", "c", "d", "e"]
        with pytest.warns(Warning):
            df = load_pandas_df(
                size=size,
                local_cache_path=tmp_dir,
                header=header,
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

        # Test if raw-zip file, rating file, and item file are cached
        assert len(os.listdir(tmp_dir)) == 3

    # Test default arguments
    df = load_pandas_df(size)
    assert len(df) == num_samples
    assert len(df.columns) == 4


@pytest.mark.smoke
@pytest.mark.parametrize(
    "size, num_movies, movie_example, title_example, genres_example, year_example",
    [("100k", 1682, 1, "Toy Story (1995)", "Animation|Children's|Comedy", "1995")],
)
def test_load_item_df(
    size, num_movies, movie_example, title_example, genres_example, year_example
):
    """Test movielens item data load (not rating data)
    """
    with TemporaryDirectory() as tmp_dir:
        df = load_item_df(
            size, local_cache_path=tmp_dir, movie_col=None, title_col="title"
        )
        assert len(df) == num_movies
        assert len(df.columns) == 1  # Only title column should be loaded
        assert df["title"][0] == title_example

        # Test title and genres
        df = load_item_df(
            size, local_cache_path=tmp_dir, movie_col="item", genres_col="genres"
        )
        assert len(df) == num_movies
        assert len(df.columns) == 2  # movile_col and genres_col
        assert df["item"][0] == movie_example
        assert df["genres"][0] == genres_example

        # Test release year
        df = load_item_df(size, local_cache_path=tmp_dir, year_col="year")
        assert len(df) == num_movies
        assert len(df.columns) == 2  # movile_col (default) and year_col
        assert df["year"][0] == year_example


@pytest.mark.smoke
@pytest.mark.spark
@pytest.mark.parametrize(
    "size, num_samples, num_movies, movie_example, title_example, genres_example, year_example",
    [
        (
            "100k",
            100000,
            1682,
            1,
            "Toy Story (1995)",
            "Animation|Children's|Comedy",
            "1995",
        )
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
):
    """Test MovieLens dataset load into pySpark.DataFrame
    """
    spark = start_or_get_spark("MovieLensLoaderTesting")

    # Test if correct data are loaded and local_cache_path works
    with TemporaryDirectory() as tmp_dir:
        # Test if can handle different size of header columns
        header = ["1", "2"]
        schema = StructType([StructField("u", IntegerType())])
        with pytest.warns(Warning):
            # Test if schema is used when both schema and header are provided
            df = load_spark_df(
                spark, size=size, local_cache_path=tmp_dir, header=header, schema=schema
            )
            assert df.count() == num_samples
            assert len(df.columns) == len(schema)

        # Test title, genres, and released year load
        header = ["a", "b", "c", "d", "e"]
        with pytest.warns(Warning):
            df = load_spark_df(
                spark,
                size=size,
                local_cache_path=tmp_dir,
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

        # Test if raw-zip file, rating file, and item file are cached
        assert len(os.listdir(tmp_dir)) == 3

    # Test default arguments
    df = load_spark_df(spark, size)
    assert df.count() == num_samples
    assert len(df.columns) == 4


@pytest.mark.smoke
@pytest.mark.parametrize("size", ["100k"])
def test_download_and_extract_movielens(size):
    """Test movielens data download and extract
    """
    with TemporaryDirectory() as tmp_dir:
        zip_path = os.path.join(tmp_dir, "ml.zip")
        download_movielens(size, dest_path=zip_path)
        assert len(os.listdir(tmp_dir)) == 1
        assert os.path.exists(zip_path)

        rating_path = os.path.join(tmp_dir, "rating.dat")
        item_path = os.path.join(tmp_dir, "item.dat")
        extract_movielens(
            size, rating_path=rating_path, item_path=item_path, zip_path=zip_path
        )
        assert len(os.listdir(tmp_dir)) == 3
        assert os.path.exists(rating_path)
        assert os.path.exists(item_path)
