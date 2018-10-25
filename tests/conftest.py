import pytest
import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
import logging

# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
    return (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


@pytest.fixture(scope="module")
def sar_settings():
    return {
        # absolute tolerance parameter for matrix equivalence in SAR tests
        "ATOL": 1e-1,
        # directory of the current file - used to link unit test data
        "FILE_DIR": "http://recodatasets.blob.core.windows.net/sarunittest/",
        # user ID used in the test files (they are designed for this user ID, this is part of the test)
        "TEST_USER_ID": "0003000098E85347",
    }


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
def pandas_dummy(header):
    ratings_dict = {
        header["col_user"]: [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        header["col_item"]: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        header["col_rating"]: [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    }
    df = pd.DataFrame(ratings_dict)
    return df


@pytest.fixture(scope="module")
def pandas_dummy_timestamp(pandas_dummy, header):
    time = 1535133442
    time_series = [time + 20 * i for i in range(10)]
    df = pandas_dummy
    df[header["col_timestamp"]] = time_series
    return df


@pytest.fixture(scope="module")
def train_test_dummy_timestamp(pandas_dummy_timestamp):
    return train_test_split(pandas_dummy_timestamp, test_size=0.2, random_state=0)


@pytest.fixture(scope="module")
def demo_usage_data(header, sar_settings):
    # load the data
    data = pd.read_csv(sar_settings["FILE_DIR"] + "demoUsage.csv")
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


@pytest.fixture
def demo_usage_data_spark(spark, demo_usage_data, header):
    data_local = demo_usage_data[[x[1] for x in header.items()]]
    # TODO: install pyArrow in DS VM
    # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    data = spark.createDataFrame(data_local)
    return data
