# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Mock dataset schema to generate fake data for testing use. This will mimic the Movielens Dataset
"""
try:
    import pandera as pa
except ImportError as e:
    raise ImportError("Pandera not installed. Try `pip install recommender['dev']`") from e

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_GENRE_COL,
    DEFAULT_HEADER
)

import random
from typing import Optional

import pandas
import pyspark.sql
from pandera.typing import Series
from pandera import Field
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType


class MockMovielensSchema(pa.SchemaModel):
    """
    Mock dataset schema to generate fake data for testing purpose.
    This schema is configured to mimic the Movielens dataset

    http://files.grouplens.org/datasets/movielens/ml-100k/
    """
    # The 100k dataset has 943 total users
    userID: Series[int] = Field(in_range={"min_value": 1, "max_value": 10})
    # And 1682 total items
    itemID: Series[int] = Field(in_range={"min_value": 1, "max_value": 10})
    # Rating is on the scale from 1 to 5
    rating: Series[float] = Field(in_range={"min_value": 1, "max_value": 5})
    timestamp: Series[str] = Field(eq="2022-2-22")
    title: Series[str] = Field(eq="foo")
    genre: Series[str] = Field(eq="genreA|0")

    @classmethod
    def get_df(
        cls,
        size: int = 3, seed: int = 100,
        keep_first_n_cols: Optional[int] = None,
        keep_title_col: bool = False, keep_genre_col: bool = False,
    ) -> pandas.DataFrame:
        """Return fake movielens dataset as a Pandas Dataframe with specified rows.

        Args:
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            keep_first_n_cols (int, optional): keep the first n default movielens columns.
            keep_title_col (bool): remove the title column if False. Defaults to True.
            keep_genre_col (bool): remove the genre column if False. Defaults to True.

        Returns:
            pandas.DataFrame: a mock dataset
        """
        schema = cls.to_schema()
        if keep_first_n_cols is not None:
            if keep_first_n_cols < 1 or keep_first_n_cols > len(DEFAULT_HEADER):
                raise ValueError(f"Invalid value for 'keep_first_n_cols': {keep_first_n_cols}. Valid range: [1-{len(DEFAULT_HEADER)}]")
            schema = schema.remove_columns(DEFAULT_HEADER[keep_first_n_cols:])
        if not keep_title_col:
            schema = schema.remove_columns([DEFAULT_TITLE_COL])
        if not keep_genre_col:
            schema = schema.remove_columns([DEFAULT_GENRE_COL])

        random.seed(seed)
        return schema.example(size=size)

    @classmethod
    def get_spark_df(
        cls,
        spark: SparkSession,
        size: int = 3, seed: int = 100,
        keep_title_col: bool = False, keep_genre_col: bool = False,
    ) -> pyspark.sql.DataFrame:
        """Return fake movielens dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            keep_title_col (bool): remove the title column if False. Defaults to False.
            keep_genre_col (bool): remove the genre column if False. Defaults to False.

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(size=size, seed=seed, keep_title_col=True, keep_genre_col=True)
        # serialize the pandas.df to avoid the expensive java <-> python communication
        pandas_df.to_csv('test.csv', header=False, index=False)

        deserialization_schema = StructType([
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, StringType()),
            StructField(DEFAULT_TITLE_COL, StringType()),
            StructField(DEFAULT_GENRE_COL, StringType()),
        ])
        spark_df = spark.read.csv('test.csv', schema=deserialization_schema)

        if not keep_title_col:
            spark_df = spark_df.drop(DEFAULT_TITLE_COL)
        if not keep_genre_col:
            spark_df = spark_df.drop(DEFAULT_GENRE_COL)
        return spark_df
