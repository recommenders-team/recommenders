import csv
import urllib.request
import codecs
import pytest
import pandas as pd

try:
    from pyspark.sql import SparkSession
except:
    pass


@pytest.fixture(scope="session")
def start_spark_test(app_name="Sample", url="local[*]", memory="1G"):
    """Start Spark if not started
    Args:
        app_name (str): sets name of the application
        url (str): url for spark master
        memory (str): size of memory for spark driver
    """

    """
    Other Spark settings which you might find useful:
        .config("spark.executor.cores", "4")
        .config("spark.executor.memory", "2g")
        .config("spark.memory.fraction", "0.9")
        .config("spark.memory.stageFraction", "0.3")
        .config("spark.executor.instances", 1)
        .config("spark.executor.heartbeatInterval", "36000s")
        .config("spark.network.timeout", "10000000s")
        .config("spark.driver.maxResultSize", memory)
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )

    return spark


@pytest.fixture(scope="module")
def header():
    header = {
        "col_user": "UserId",
        "col_item": "MovieId",
        "col_rating": "Rating",
        "col_timestamp": "Timestamp",
    }
    return header


@pytest.fixture(scope="module")
def load_pandas_dummy_dataset():
    """Load sample dataset in pandas for testing; can be used to create a Spark dataframe
    Returns:
        single Pandas dataframe
    """
    ratings_dict = {
        header()["col_user"]: [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        header()["col_item"]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        header()["col_rating"]: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    }
    dataframe = pd.DataFrame(ratings_dict)

    return dataframe


@pytest.fixture(scope="module")
def load_pandas_dummy_timestamp_dataset():
    """Load sample dataset in pandas for testing; can be used to create a Spark dataframe
       This method adds an additional column.
    Returns:
        single Pandas dataframe
    """
    time = 1535133442
    time_series = pd.Series([time] * 10)
    dataframe = load_pandas_dummy_dataset()
    dataframe[header()["col_timestamp"]] = time_series.values

    return dataframe


@pytest.fixture(scope="module")
def csv_reader_url(url, delimiter=",", encoding="utf-8"):
    """
    Read a csv file over http

    Returns:
         csv reader iterable
    """
    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, encoding), delimiter=delimiter)
    return csvfile
