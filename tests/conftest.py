# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don't need to import the module you defined your fixtures to use in a test,
# it automatically gets discovered by pytest and thus you can simply receive fixture objects by naming them as
# an input argument in the test."

import calendar
import datetime
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.utils.spark_utils import start_or_get_spark


@pytest.fixture(scope="session")
def output_notebook():
    return "output.ipynb"


@pytest.fixture(scope="session")
def kernel_name():
    """Unless manually modified, python3 should be the name of the current jupyter kernel
    that runs on the activated conda environment"""
    return "python3"


def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.path.pardir, "examples")
    )


@pytest.fixture
def tmp(tmp_path_factory):
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


@pytest.fixture(scope="session")
def spark(tmp_path_factory, app_name="Sample", url="local[*]"):
    """Start Spark if not started.

    Other Spark settings which you might find useful:
        .config("spark.executor.cores", "4")
        .config("spark.executor.memory", "2g")
        .config("spark.memory.fraction", "0.9")
        .config("spark.memory.stageFraction", "0.3")
        .config("spark.executor.instances", 1)
        .config("spark.executor.heartbeatInterval", "36000s")
        .config("spark.network.timeout", "10000000s")

    Args:
        app_name (str): sets name of the application
        url (str): url for spark master

    Returns:
        SparkSession: new Spark session
    """

    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        config = {
            "spark.local.dir": td,
            "spark.sql.shuffle.partitions": 1,
            "spark.sql.crossJoin.enabled": "true",
        }
        spark = start_or_get_spark(app_name=app_name, url=url, config=config)
        yield spark
        spark.stop()


@pytest.fixture(scope="module")
def sar_settings():
    return {
        # absolute tolerance parameter for matrix equivalence in SAR tests
        "ATOL": 1e-8,
        # directory of the current file - used to link unit test data
        "FILE_DIR": "https://recodatasets.z20.web.core.windows.net/sarunittest/",
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
        header["col_rating"]: [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0],
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
    data["rating"] = pd.Series([1.0] * data.shape[0])
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
        lambda s: float(
            calendar.timegm(
                datetime.datetime.strptime(s, "%Y/%m/%dT%H:%M:%S").timetuple()
            )
        )
    )

    return data


@pytest.fixture(scope="module")
def demo_usage_data_spark(spark, demo_usage_data, header):
    data_local = demo_usage_data[[x[1] for x in header.items()]]
    return spark.createDataFrame(data_local)


@pytest.fixture(scope="module")
def criteo_first_row():
    return {
        "label": 0,
        "int00": 1,
        "int01": 1,
        "int02": 5,
        "int03": 0,
        "int04": 1382,
        "int05": 4,
        "int06": 15,
        "int07": 2,
        "int08": 181,
        "int09": 1,
        "int10": 2,
        "int11": None,
        "int12": 2,
        "cat00": "68fd1e64",
        "cat01": "80e26c9b",
        "cat02": "fb936136",
        "cat03": "7b4723c4",
        "cat04": "25c83c98",
        "cat05": "7e0ccccf",
        "cat06": "de7995b8",
        "cat07": "1f89b562",
        "cat08": "a73ee510",
        "cat09": "a8cd5504",
        "cat10": "b2cb9c98",
        "cat11": "37c9c164",
        "cat12": "2824a5f6",
        "cat13": "1adce6ef",
        "cat14": "8ba8b39a",
        "cat15": "891b62e7",
        "cat16": "e5ba7672",
        "cat17": "f54016b9",
        "cat18": "21ddcdc9",
        "cat19": "b1252a9d",
        "cat20": "07b5194c",
        "cat21": None,
        "cat22": "3a171ecb",
        "cat23": "c5c50484",
        "cat24": "e8b83407",
        "cat25": "9727dd16",
    }


@pytest.fixture(scope="module")
def notebooks():
    folder_notebooks = path_notebooks()

    # Path for the notebooks
    paths = {
        "template": os.path.join(folder_notebooks, "template.ipynb"),
        "sar_single_node": os.path.join(
            folder_notebooks, "00_quick_start", "sar_movielens.ipynb"
        ),
        "ncf": os.path.join(folder_notebooks, "00_quick_start", "ncf_movielens.ipynb"),
        "als_pyspark": os.path.join(
            folder_notebooks, "00_quick_start", "als_movielens.ipynb"
        ),
        "fastai": os.path.join(
            folder_notebooks, "00_quick_start", "fastai_movielens.ipynb"
        ),
        "xdeepfm_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "xdeepfm_criteo.ipynb"
        ),
        "dkn_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "dkn_MIND.ipynb"
        ),
        "lightgbm_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "lightgbm_tinycriteo.ipynb"
        ),
        "wide_deep": os.path.join(
            folder_notebooks, "00_quick_start", "wide_deep_movielens.ipynb"
        ),
        "slirec_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "sequential_recsys_amazondataset.ipynb"
        ),
        "nrms_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "nrms_MIND.ipynb"
        ),
        "naml_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "naml_MIND.ipynb"
        ),
        "lstur_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "lstur_MIND.ipynb"
        ),
        "npa_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "npa_MIND.ipynb"
        ),
        "rlrmc_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "rlrmc_movielens.ipynb"
        ),
        "geoimc_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "geoimc_movielens.ipynb"
        ),
        "sasrec_quickstart": os.path.join(
            folder_notebooks, "00_quick_start", "sasrec_amazon.ipynb"
        ),
        "data_split": os.path.join(
            folder_notebooks, "01_prepare_data", "data_split.ipynb"
        ),
        "wikidata_knowledge_graph": os.path.join(
            folder_notebooks, "01_prepare_data", "wikidata_knowledge_graph.ipynb"
        ),
        "mind_utils": os.path.join(
            folder_notebooks, "01_prepare_data", "mind_utils.ipynb"
        ),
        "als_deep_dive": os.path.join(
            folder_notebooks, "02_model_collaborative_filtering", "als_deep_dive.ipynb"
        ),
        "surprise_svd_deep_dive": os.path.join(
            folder_notebooks,
            "02_model_collaborative_filtering",
            "surprise_svd_deep_dive.ipynb",
        ),
        "baseline_deep_dive": os.path.join(
            folder_notebooks,
            "02_model_collaborative_filtering",
            "baseline_deep_dive.ipynb",
        ),
        "lightgcn_deep_dive": os.path.join(
            folder_notebooks,
            "02_model_collaborative_filtering",
            "lightgcn_deep_dive.ipynb",
        ),
        "ncf_deep_dive": os.path.join(
            folder_notebooks, "02_model_hybrid", "ncf_deep_dive.ipynb"
        ),
        "sar_deep_dive": os.path.join(
            folder_notebooks, "02_model_collaborative_filtering", "sar_deep_dive.ipynb"
        ),
        "vowpal_wabbit_deep_dive": os.path.join(
            folder_notebooks,
            "02_model_content_based_filtering",
            "vowpal_wabbit_deep_dive.ipynb",
        ),
        "mmlspark_lightgbm_criteo": os.path.join(
            folder_notebooks,
            "02_model_content_based_filtering",
            "mmlspark_lightgbm_criteo.ipynb",
        ),
        "cornac_bpr_deep_dive": os.path.join(
            folder_notebooks,
            "02_model_collaborative_filtering",
            "cornac_bpr_deep_dive.ipynb",
        ),
        "cornac_bivae_deep_dive": os.path.join(
            folder_notebooks,
            "02_model_collaborative_filtering",
            "cornac_bivae_deep_dive.ipynb",
        ),
        "xlearn_fm_deep_dive": os.path.join(
            folder_notebooks, "02_model_hybrid", "fm_deep_dive.ipynb"
        ),
        "evaluation": os.path.join(folder_notebooks, "03_evaluate", "evaluation.ipynb"),
        "evaluation_diversity": os.path.join(
            folder_notebooks, "03_evaluate", "als_movielens_diversity_metrics.ipynb"
        ),
        "spark_tuning": os.path.join(
            folder_notebooks, "04_model_select_and_optimize", "tuning_spark_als.ipynb"
        ),
        "nni_tuning_svd": os.path.join(
            folder_notebooks, "04_model_select_and_optimize", "nni_surprise_svd.ipynb"
        ),
    }
    return paths


# NCF FIXTURES


@pytest.fixture(scope="module")
def test_specs_ncf():
    return {
        "number_of_rows": 1000,
        "user_ids": [1, 2, 3, 4, 5],
        "seed": 123,
        "ratio": 0.6,
        "split_numbers": [2, 3, 5],
        "tolerance": 0.01,
    }


@pytest.fixture(scope="module")
def dataset_ncf(test_specs_ncf):
    """Get Python labels"""

    def random_date_generator(start_date, range_in_days):
        """Helper function to generate random timestamps.

        Reference: https://stackoverflow.com/questions/41006182/generate-random-dates-within-a-range-in-numpy
        """
        days_to_add = np.arange(0, range_in_days)
        random_dates = []
        for i in range(range_in_days):
            random_date = np.datetime64(start_date) + np.random.choice(days_to_add)
            random_dates.append(random_date)

        return random_dates

    np.random.seed(test_specs_ncf["seed"])

    rating = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.randint(
                1, 100, test_specs_ncf["number_of_rows"]
            ),
            DEFAULT_ITEM_COL: np.random.randint(
                1, 100, test_specs_ncf["number_of_rows"]
            ),
            DEFAULT_RATING_COL: np.random.randint(
                1, 5, test_specs_ncf["number_of_rows"]
            ),
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01", test_specs_ncf["number_of_rows"]
            ),
        }
    )

    train, test = python_chrono_split(rating, ratio=test_specs_ncf["ratio"])

    return train, test


@pytest.fixture
def dataset_ncf_files(dataset_ncf):
    train, test = dataset_ncf
    test = test[test["userID"].isin(train["userID"].unique())]
    test = test[test["itemID"].isin(train["itemID"].unique())]
    train = train.sort_values(by=DEFAULT_USER_COL)
    test = test.sort_values(by=DEFAULT_USER_COL)
    leave_one_out_test = test.groupby("userID").last().reset_index()
    return train, test, leave_one_out_test


@pytest.fixture
def data_paths(tmp_path):
    train_path = os.path.join(tmp_path, "train.csv")
    test_path = os.path.join(tmp_path, "test.csv")
    leave_one_out_test_path = os.path.join(tmp_path, "leave_one_out_test.csv")
    return train_path, test_path, leave_one_out_test_path


@pytest.fixture
def dataset_ncf_files_sorted(data_paths, dataset_ncf_files):
    train_path, test_path, leave_one_out_test_path = data_paths
    train, test, leave_one_out_test = dataset_ncf_files
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_path, index=False)
    return train_path, test_path, leave_one_out_test_path


@pytest.fixture
def dataset_ncf_files_unsorted(data_paths, dataset_ncf_files):
    train_path, test_path, leave_one_out_test_path = data_paths
    train, test, leave_one_out_test = dataset_ncf_files
    # shift last row to the first
    train = train.apply(np.roll, shift=1)
    test = test.apply(np.roll, shift=1)
    leave_one_out_test = leave_one_out_test.apply(np.roll, shift=1)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_path, index=False)
    return train_path, test_path, leave_one_out_test_path


@pytest.fixture
def dataset_ncf_files_empty(data_paths, dataset_ncf_files):
    train_path, test_path, leave_one_out_test_path = data_paths
    train, test, leave_one_out_test = dataset_ncf_files
    train = train[0:0]
    test = test[0:0]
    leave_one_out_test = leave_one_out_test[0:0]
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_path, index=False)
    return train_path, test_path, leave_one_out_test_path


@pytest.fixture
def dataset_ncf_files_missing_column(data_paths, dataset_ncf_files):
    train_path, test_path, leave_one_out_test_path = data_paths
    train, test, leave_one_out_test = dataset_ncf_files
    train = train.drop(DEFAULT_USER_COL, axis=1)
    test = test.drop(DEFAULT_USER_COL, axis=1)
    leave_one_out_test = leave_one_out_test.drop(DEFAULT_USER_COL, axis=1)
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    leave_one_out_test.to_csv(leave_one_out_test_path, index=False)
    return train_path, test_path, leave_one_out_test_path


# RBM Fixtures


@pytest.fixture(scope="module")
def test_specs():
    return {
        "users": 30,
        "items": 53,
        "ratings": 5,
        "seed": 123,
        "spars": 0.8,
        "ratio": 0.7,
    }


@pytest.fixture(scope="module")
def affinity_matrix(test_specs):
    """Generate a random user/item affinity matrix. By increasing the likehood of 0 elements we simulate
    a typical recommending situation where the input matrix is highly sparse.

    Args:
        users (int): number of users (rows).
        items (int): number of items (columns).
        ratings (int): rating scale, e.g. 5 meaning rates are from 1 to 5.
        spars: probability of obtaining zero. This roughly corresponds to the sparseness.
               of the generated matrix. If spars = 0 then the affinity matrix is dense.

    Returns:
        np.array: sparse user/affinity matrix of integers.

    """

    np.random.seed(test_specs["seed"])

    # uniform probability for the 5 ratings
    s = [(1 - test_specs["spars"]) / test_specs["ratings"]] * test_specs["ratings"]
    s.append(test_specs["spars"])
    P = s[::-1]

    # generates the user/item affinity matrix. Ratings are from 1 to 5, with 0s denoting unrated items
    X = np.random.choice(
        test_specs["ratings"] + 1, (test_specs["users"], test_specs["items"]), p=P
    )

    Xtr, Xtst = numpy_stratified_split(
        X, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    return (Xtr, Xtst)


# DeepRec Fixtures


@pytest.fixture(scope="session")
def deeprec_resource_path():
    return Path(__file__).absolute().parent.joinpath("resources", "deeprec")


@pytest.fixture(scope="session")
def mind_resource_path(deeprec_resource_path):
    return Path(__file__).absolute().parent.joinpath("resources", "mind")


@pytest.fixture(scope="module")
def deeprec_config_path():
    return (
        Path(__file__)
        .absolute()
        .parents[1]
        .joinpath("recommenders", "models", "deeprec", "config")
    )
