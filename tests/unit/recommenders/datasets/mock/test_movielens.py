from recommenders.datasets.mock.movielens import MockMovielensSchema
from recommenders.datasets.movielens import DEFAULT_HEADER
from recommenders.utils.constants import (
    DEFAULT_GENRE_COL,
    DEFAULT_TITLE_COL,
)

import pytest
import pandas
import pyspark.sql
from pyspark.sql import SparkSession


@pytest.mark.parametrize("size", [10, 100])
def test_mock_movielens_schema__has_default_col_names(size):
    df = MockMovielensSchema.example(size=size)
    for col_name in DEFAULT_HEADER:
        assert col_name in df.columns


@pytest.mark.parametrize("keep_first_n_cols", [1, 2, 3, 4])
def test_mock_movielens_schema__get_df_remove_default_col__return_success(keep_first_n_cols):
    df = MockMovielensSchema.get_df(size=3, keep_first_n_cols=keep_first_n_cols)
    assert len(df) > 0
    assert len(df.columns) == keep_first_n_cols


@pytest.mark.parametrize("keep_first_n_cols", [-1, 0, 100])
def test_mock_movielens_schema__get_df_invalid_param__return_failure(keep_first_n_cols):
    with pytest.raises(ValueError, match=r"Invalid value.*"):
        MockMovielensSchema.get_df(size=3, keep_first_n_cols=keep_first_n_cols)


@pytest.mark.parametrize("keep_genre_col", [True, False])
@pytest.mark.parametrize("keep_title_col", [True, False])
@pytest.mark.parametrize("keep_first_n_cols", [None, 2])
@pytest.mark.parametrize("seed", [-1])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [0, 3, 10])
def test_mock_movielens_schema__get_df__return_success(size, seed, keep_first_n_cols, keep_title_col, keep_genre_col):
    df = MockMovielensSchema.get_df(
        size=size, seed=seed,
        keep_first_n_cols=keep_first_n_cols,
        keep_title_col=keep_title_col, keep_genre_col=keep_genre_col
    )
    assert type(df) == pandas.DataFrame
    assert len(df) == size

    if keep_title_col:
        assert len(df[DEFAULT_TITLE_COL]) == size
    if keep_genre_col:
        assert len(df[DEFAULT_GENRE_COL]) == size


@pytest.mark.parametrize("keep_genre_col", [True, False])
@pytest.mark.parametrize("keep_title_col", [True, False])
@pytest.mark.parametrize("seed", [101])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [0, 3, 10])
def test_mock_movielens_schema__get_spark_df__return_success(spark: SparkSession, size, seed, keep_title_col, keep_genre_col):
    df = MockMovielensSchema.get_spark_df(spark, size=size, seed=seed, keep_title_col=keep_title_col, keep_genre_col=keep_genre_col)
    assert type(df) == pyspark.sql.DataFrame
    assert df.count() == size

    if keep_title_col:
        assert df.schema[DEFAULT_TITLE_COL]
    if keep_genre_col:
        assert df.schema[DEFAULT_GENRE_COL]
