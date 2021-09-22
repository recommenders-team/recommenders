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
)

import random
from typing import Optional

from pandera.typing import DateTime, Series
from pandera import Field, Check
from pandera.schemas import DataFrameSchema
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, LongType, IntegerType, StringType, FloatType


class MockMovielens100kSchema(pa.SchemaModel):
    """
    Mock dataset schema to generate fake data for testing purpose.
    This schema is configured to mimic the Movielens 100k dataset

    http://files.grouplens.org/datasets/movielens/ml-100k/
    """
    # The 100k dataset has 943 total users
    userID: Series[int] = Field(in_range={"min_value": 1, "max_value": 10})
    # And 1682 total items
    itemID: Series[int] = Field(in_range={"min_value": 1, "max_value": 10})
    # Rating is on the scale from 1 to 5
    rating: Series[int] = Field(in_range={"min_value": 1, "max_value": 5})
    timestamp: Series[int]
    title: Series[str] = Field(eq="foo")
    genres: Series[str] = Field(eq="genreA|0")

    @classmethod
    def get_df(
        cls,
        size: int = 3, seed: int = 100,
        # title_col: Optional[str] = None, genres_col: Optional[str] = None
    ):
        """Return fake movielens dataset as a Pandas Dataframe with specified rows.

        Args:
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            title_col (str, optional): if not None, append a title column. Defaults to None.
            genres_col (str, optional): if not None, append a genre column. Defaults to None.

        Returns:
            pandas.DataFrame: a mock dataset
        """
        random.seed(seed)
        return cls.example(size=size)

    @classmethod
    def get_spark_df(
        cls,
        spark: SparkSession,
        size: int = 3, seed: int = 100,
        # title_col: Optional[str] = None, genres_col: Optional[str] = None,
        # schema: Optional[StructType] = None
    ):
        """Return fake movielens dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            title_col (str, optional): if not None, append a title column. Defaults to None.
            genres_col (str, optional): if not None, append a genre column. Defaults to None.
            schema (pyspark.sql.types.StructType, optional): dataset schema. Defaults to None.

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(size=size, seed=seed)
        pandas_df.to_csv('test.csv', header=False, index=False)
        default_schema = StructType([
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, LongType()),
            StructField("title", StringType()),
            StructField("genres", StringType()),
        ])
        return spark.read.csv('test.csv', schema=default_schema)

    # @classmethod
    # def _get_item_df(cls, size, title_col: Optional[str] = None, genres_col: Optional[str] = None):
    #     schema = DataFrameSchema()  # create an empty schema
    #     if title_col is not None:
    #         # adds a title column with random alphabets
    #         schema = schema.add_columns({title_col: pa.Column(str, Check.str_matches(r'^[a-z]+$'))})
    #     if genres_col is not None:
    #         # adds a genre column with '|' separated string
    #         schema = schema.add_columns({genres_col: pa.Column(str, Check.str_matches(r'^[a-z]+\|[0-9]$'))})
    #     schema.example()