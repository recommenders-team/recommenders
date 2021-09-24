# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Mock dataset schema to generate fake data for testing use. This will mimic the Movielens Dataset
"""
try:
    import pandera as pa
except ImportError:
    raise ImportError("pandera is not installed. Try `pip install recommenders['dev']`")

try:
    from pyspark.sql.types import StructField, StructType, IntegerType, StringType, FloatType
except ImportError:
    pass  # so the environment without spark doesn't break

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_TITLE_COL,
    DEFAULT_GENRE_COL,
    DEFAULT_HEADER
)
from recommenders.datasets.download_utils import download_path

import os
import random
from typing import Optional

import pandas
from pandera.typing import Series
from pandera import Field


class MockMovielensSchema(pa.SchemaModel):
    """
    Mock dataset schema to generate fake data for testing purpose.
    This schema is configured to mimic the Movielens dataset

    http://files.grouplens.org/datasets/movielens/ml-100k/

    Dataset schema and generation is configured using pandera.
    Please see https://pandera.readthedocs.io/en/latest/schema_models.html
    for more information.
    """
    # Some notebooks will do a cross join with userID and itemID,
    # a sparse range for these IDs can slow down the notebook tests
    userID: Series[int] = Field(in_range={"min_value": 1, "max_value": 10})
    itemID: Series[int] = Field(in_range={"min_value": 1, "max_value": 10})
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
        # For more information on data synthesis, see https://pandera.readthedocs.io/en/latest/data_synthesis_strategies.html
        return schema.example(size=size)

    @classmethod
    def get_spark_df(
        cls,
        spark,
        size: int = 3, seed: int = 100,
        keep_title_col: bool = False, keep_genre_col: bool = False,
        tmp_path: Optional[str] = None,
    ):
        """Return fake movielens dataset as a Spark Dataframe with specified rows

        Args:
            spark (SparkSession): spark session to load the dataframe into
            size (int): number of rows to generate
            seed (int): seeding the pseudo-number generation. Defaults to 100.
            keep_title_col (bool): remove the title column if False. Defaults to False.
            keep_genre_col (bool): remove the genre column if False. Defaults to False.
            tmp_path (str, optional): path to store files for serialization purpose
                when transferring data from python to java.
                If None, a temporal path is used instead

        Returns:
            pyspark.sql.DataFrame: a mock dataset
        """
        pandas_df = cls.get_df(size=size, seed=seed, keep_title_col=True, keep_genre_col=True)

        # generate temp folder
        with download_path(tmp_path) as tmp_folder:
            filepath = os.path.join(tmp_folder, f"mock_movielens_{size}.csv")
            # serialize the pandas.df as a csv to avoid the expensive java <-> python communication
            pandas_df.to_csv(filepath, header=False, index=False)
            print(f"Saving file {filepath}.")
            spark_df = spark.read.csv(filepath, schema=cls._get_spark_deserialization_schema())
            # Cache and force trigger action since data-file might be removed.
            spark_df.cache()
            spark_df.count()

        if not keep_title_col:
            spark_df = spark_df.drop(DEFAULT_TITLE_COL)
        if not keep_genre_col:
            spark_df = spark_df.drop(DEFAULT_GENRE_COL)
        return spark_df

    @classmethod
    def _get_spark_deserialization_schema(cls):
        return StructType([
            StructField(DEFAULT_USER_COL, IntegerType()),
            StructField(DEFAULT_ITEM_COL, IntegerType()),
            StructField(DEFAULT_RATING_COL, FloatType()),
            StructField(DEFAULT_TIMESTAMP_COL, StringType()),
            StructField(DEFAULT_TITLE_COL, StringType()),
            StructField(DEFAULT_GENRE_COL, StringType()),
        ])
