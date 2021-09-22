# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Mock dataset schema to generate fake data for testing use. This will mimic the Movielens Dataset
"""
try:
    import pandera as pa
except ImportError as e:
    raise ImportError("Pandera not installed. Try `pip install recommender['dev']`") from e

import random
from typing import Optional

from pandera.typing import DateTime, Series
from pandera import Field
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType


class MockMovielens100kSchema(pa.SchemaModel):
    """
    Mock dataset schema to generate fake data for testing purpose.
    This schema is configured to mimic the Movielens 100k dataset

    http://files.grouplens.org/datasets/movielens/ml-100k/
    """
    # The 100k dataset has 943 total users
    userID: Series[int] = Field(in_range={"min_value": 1, "max_value": 943})
    # And 1682 total items
    itemID: Series[int] = Field(in_range={"min_value": 1, "max_value": 1682})
    # Rating is on the scale from 1 to 5
    rating: Series[int] = Field(in_range={"min_value": 1, "max_value": 5})
    timestamp: Series[DateTime]
    title: Series[str] = Field(eq="foo")
    genres: Series[str] = Field(eq="genreA|0")

    @classmethod
    def get_df(cls, size: int = 3, seed: int = 100):
        """Return fake movielens dataset as a Pandas Dataframe with specified rows.

        Args:
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.

        Returns:
            pandas.DataFrame: a mock dataset
        """
        random.seed(seed)
        return cls.example(size=size)

    @classmethod
    def get_spark_df(cls, spark: SparkSession, size: int = 3, seed: int = 100, schema: Optional[StructType] = None):
        """Return fake movielens dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int, optional): seeding the pseudo-number generation. Defaults to 100.
            schema (pyspark.sql.types.StructType optional): [description]. Defaults to None.

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(size=size, seed=seed)
        return spark.createDataFrame(pandas_df, schema=schema)
