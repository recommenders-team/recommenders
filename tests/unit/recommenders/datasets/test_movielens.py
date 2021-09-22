from recommenders.datasets.movielens import DATA_FORMAT, MOCK_DATA_FORMAT
from recommenders.datasets.movielens import load_pandas_df, load_spark_df

import pyspark.sql
from pyspark.sql import SparkSession


def test_mock_movielens_data__no_name_collision():
    """
    Making sure that no common names are shared between the mock and real dataset sizes
    """
    dataset_name = set(DATA_FORMAT.keys())
    dataset_name_mock = set(MOCK_DATA_FORMAT.keys())
    collision = dataset_name.intersection(dataset_name_mock)
    assert not collision


def test_mock_movielens_data_generation_succeed(spark: SparkSession):
    df = load_spark_df(spark, "mock100")
    assert type(df) == pyspark.sql.DataFrame
    assert df.count() == 100
