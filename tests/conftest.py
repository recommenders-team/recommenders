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
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from tempfile import TemporaryDirectory
from tests.notebooks_common import path_notebooks
from reco_utils.common.spark_utils import start_or_get_spark


@pytest.fixture
def tmp(tmp_path_factory):
    with TemporaryDirectory(dir=tmp_path_factory.getbasetemp()) as td:
        yield td


@pytest.fixture(scope="session")
def spark(app_name="Sample", url="local[*]"):
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

    config = {"spark.local.dir": "/mnt", "spark.sql.shuffle.partitions": 1}
    spark = start_or_get_spark(app_name=app_name, url=url, config=config)
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def sar_settings():
    return {
        # absolute tolerance parameter for matrix equivalence in SAR tests
        "ATOL": 1e-8,
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
            folder_notebooks, "00_quick_start", "dkn_MIND_dataset.ipynb"
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
        "data_split": os.path.join(
            folder_notebooks, "01_prepare_data", "data_split.ipynb"
        ),
        "wikidata_knowledge_graph": os.path.join(
            folder_notebooks, "01_prepare_data", "wikidata_knowledge_graph.ipynb"
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
        "xlearn_fm_deep_dive": os.path.join(
            folder_notebooks, "02_model_hybrid", "fm_deep_dive.ipynb"
        ),
        "lightgcn_deep_dive": os.path.join(
            folder_notebooks, "02_model", "lightgcn_deep_dive.ipynb"
        ),
        "evaluation": os.path.join(folder_notebooks, "03_evaluate", "evaluation.ipynb"),
        "spark_tuning": os.path.join(
            folder_notebooks, "04_model_select_and_optimize", "tuning_spark_als.ipynb"
        ),
        "nni_tuning_svd": os.path.join(
            folder_notebooks, "04_model_select_and_optimize", "nni_surprise_svd.ipynb"
        ),
    }
    return paths
