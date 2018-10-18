import pytest
import pandas as pd
import time
import datetime

try:
    from pyspark.sql import SparkSession
except:
    pass


@pytest.fixture(scope="session")
def spark(app_name="Sample", url="local[*]", memory="1G"):
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
def load_pandas_dummy_dataset(header):
    """Load sample dataset in pandas for testing; can be used to create a Spark dataframe
    Returns:
        single Pandas dataframe
    """
    ratings_dict = {
        header["col_user"]: [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        header["col_item"]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        header["col_rating"]: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    }
    dataframe = pd.DataFrame(ratings_dict)

    return dataframe


@pytest.fixture(scope="module")
def load_pandas_dummy_timestamp_dataset(header):
    """Load sample dataset in pandas for testing; can be used to create a Spark dataframe
       This method adds an additional column.
    Returns:
        single Pandas dataframe
    """
    time = 1535133442
    time_series = [time + 20*i for i in range(10)]
    dataframe = load_pandas_dummy_dataset(header)
    dataframe[header["col_timestamp"]] = time_series

    return dataframe


@pytest.fixture(scope="module")
def spark_test_settings():
    return {
        # absolute tolerance parameter for matrix equivalence in SAR tests
        "ATOL": 1e-1,
        # directory of the current file - used to link unit test data
        "FILE_DIR": "http://recodatasets.blob.core.windows.net/sarunittest/",
        # user ID used in the test files (they are designed for this user ID, this is part of the test)
        "TEST_USER_ID": "0003000098E85347",
    }


@pytest.fixture
def demo_usage_data(header, spark_test_settings):
    # load the data
    data = pd.read_csv(spark_test_settings["FILE_DIR"] + "demoUsage.csv")
    data["rating"] = pd.Series([1] * data.shape[0])
    data = data.rename(
        columns={
            "userId": header["col_user"],
            "productId": header["col_item"],
            "rating": header["col_rating"],
            "timestamp": header["col_timestamp"],
        }
    )

    # convert timestamp
    data[header["col_timestamp"]] = data[header["col_timestamp"]].apply(
        lambda s: time.mktime(
            datetime.datetime.strptime(s, "%Y/%m/%dT%H:%M:%S").timetuple()
        )
    )

    return data

