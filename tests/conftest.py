import csv
import urllib.request
import codecs
import pytest
import pandas as pd


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
