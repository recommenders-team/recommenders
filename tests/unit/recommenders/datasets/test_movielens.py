import os
import pandas
import pytest

from recommenders.datasets.movielens import MockMovielensSchema
from recommenders.datasets.movielens import load_pandas_df, load_spark_df
from recommenders.datasets.movielens import (
    DATA_FORMAT,
    MOCK_DATA_FORMAT,
    DEFAULT_HEADER,
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL
)
from recommenders.utils.constants import DEFAULT_GENRE_COL, DEFAULT_TITLE_COL

from pandas.core.series import Series
from pytest_mock import MockerFixture


@pytest.mark.parametrize("size", [10, 100])
def test_mock_movielens_schema__has_default_col_names(size):
    df = MockMovielensSchema.example(size=size)
    for col_name in DEFAULT_HEADER:
        assert col_name in df.columns


@pytest.mark.parametrize("keep_first_n_cols", [2, 3, 4])
def test_mock_movielens_schema__get_df_remove_default_col__return_success(
    keep_first_n_cols,
):
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
def test_mock_movielens_schema__get_df__return_success(
    size, seed, keep_first_n_cols, keep_title_col, keep_genre_col
):
    df = MockMovielensSchema.get_df(
        size=size,
        seed=seed,
        keep_first_n_cols=keep_first_n_cols,
        keep_title_col=keep_title_col,
        keep_genre_col=keep_genre_col,
    )
    assert type(df) == pandas.DataFrame
    assert len(df) == size

    if keep_title_col:
        assert len(df[DEFAULT_TITLE_COL]) == size
    if keep_genre_col:
        assert len(df[DEFAULT_GENRE_COL]) == size


@pytest.mark.spark
@pytest.mark.parametrize("keep_genre_col", [True, False])
@pytest.mark.parametrize("keep_title_col", [True, False])
@pytest.mark.parametrize("seed", [101])  # seed for pseudo-random # generation
@pytest.mark.parametrize("size", [0, 3, 10])
def test_mock_movielens_schema__get_spark_df__return_success(
    spark, size, seed, keep_title_col, keep_genre_col
):
    df = MockMovielensSchema.get_spark_df(
        spark,
        size=size,
        seed=seed,
        keep_title_col=keep_title_col,
        keep_genre_col=keep_genre_col,
    )
    assert df.count() == size

    if keep_title_col:
        assert df.schema[DEFAULT_TITLE_COL]
    if keep_genre_col:
        assert df.schema[DEFAULT_GENRE_COL]


@pytest.mark.spark
def test_mock_movielens_schema__get_spark_df__store_tmp_file(spark, tmp_path):
    data_size = 3
    MockMovielensSchema.get_spark_df(spark, size=data_size, tmp_path=tmp_path)
    assert os.path.exists(os.path.join(tmp_path, f"mock_movielens_{data_size}.csv"))


@pytest.mark.spark
def test_mock_movielens_schema__get_spark_df__data_serialization_default_param(
    spark, mocker: MockerFixture
):
    data_size = 3
    to_csv_spy = mocker.spy(pandas.DataFrame, "to_csv")

    df = MockMovielensSchema.get_spark_df(spark, size=data_size)
    # assertions
    to_csv_spy.assert_called_once()
    assert df.count() == data_size


def test_mock_movielens_data__no_name_collision():
    """
    Making sure that no common names are shared between the mock and real dataset sizes
    """
    dataset_name = set(DATA_FORMAT.keys())
    dataset_name_mock = set(MOCK_DATA_FORMAT.keys())
    collision = dataset_name.intersection(dataset_name_mock)
    assert not collision


@pytest.mark.spark
def test_load_spark_df_mock_100__with_default_param__succeed(spark):
    df = load_spark_df(spark, "mock100")
    assert df.count() == 100


def test_load_pandas_df_mock_100__with_default_param__succeed():
    df = load_pandas_df("mock100")
    assert type(df) == pandas.DataFrame
    assert len(df) == 100
    assert not df[[DEFAULT_USER_COL, DEFAULT_ITEM_COL]].duplicated().any()


@pytest.mark.spark
def test_load_spark_df_mock_100__with_custom_param__succeed(spark):
    df = load_spark_df(
        spark, "mock100", title_col=DEFAULT_TITLE_COL, genres_col=DEFAULT_GENRE_COL
    )
    assert df.schema[DEFAULT_TITLE_COL]
    assert df.schema[DEFAULT_GENRE_COL]
    assert df.count() == 100
    assert "|" in df.take(1)[0][DEFAULT_GENRE_COL]
    assert df.take(1)[0][DEFAULT_TITLE_COL] == "foo"


def test_load_pandas_df_mock_100__with_custom_param__succeed():
    df = load_pandas_df(
        "mock100", title_col=DEFAULT_TITLE_COL, genres_col=DEFAULT_GENRE_COL
    )
    assert type(df[DEFAULT_TITLE_COL]) == Series
    assert type(df[DEFAULT_GENRE_COL]) == Series
    assert len(df) == 100
    assert "|" in df.loc[0, DEFAULT_GENRE_COL]
    assert df.loc[0, DEFAULT_TITLE_COL] == "foo"
