from recommenders.datasets.movielens import DATA_FORMAT, MOCK_DATA_FORMAT
from recommenders.datasets.movielens import load_pandas_df, load_spark_df
from recommenders.utils.constants import DEFAULT_GENRE_COL, DEFAULT_TITLE_COL

import pandas
import pytest
from pandas.core.series import Series


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


@pytest.mark.spark
def test_load_spark_df_mock_100__with_custom_param__succeed(spark):
    df = load_spark_df(spark, "mock100", title_col=DEFAULT_TITLE_COL, genres_col=DEFAULT_GENRE_COL)
    assert df.schema[DEFAULT_TITLE_COL]
    assert df.schema[DEFAULT_GENRE_COL]
    assert df.count() == 100
    assert '|' in df.take(1)[0][DEFAULT_GENRE_COL]
    assert df.take(1)[0][DEFAULT_TITLE_COL] == 'foo'


def test_load_pandas_df_mock_100__with_custom_param__succeed():
    df = load_pandas_df("mock100", title_col=DEFAULT_TITLE_COL, genres_col=DEFAULT_GENRE_COL)
    assert type(df[DEFAULT_TITLE_COL]) == Series
    assert type(df[DEFAULT_GENRE_COL]) == Series
    assert len(df) == 100
    assert '|' in df.loc[0, DEFAULT_GENRE_COL]
    assert df.loc[0, DEFAULT_TITLE_COL] == 'foo'
