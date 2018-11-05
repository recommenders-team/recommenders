import math
import pandas as pd
import pytest
import os

from pyspark.sql import SparkSession

from pysarplus import SARPlus, SARModel

def assert_compare(expected_id, expected_score, actual_prediction):
    assert expected_id == actual_prediction.id
    assert math.isclose(expected_score, actual_prediction.score, rel_tol=1e-3, abs_tol=1e-3)

@pytest.fixture(scope="module")
def spark(app_name="Sample", url="local[*]", memory="1G"):
    """Start Spark if not started
    Args:
        app_name (str): sets name of the application
        url (str): url for spark master
        memory (str): size of memory for spark driver
    """

    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.jars", os.path.dirname(__file__) + "/../../scala/target/scala-2.11/sarplus_2.11-0.2.4.jar")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.crossJoin.enabled", True)
        .config("spark.ui.enabled", False)
        .getOrCreate()
    )

    return spark

@pytest.fixture(scope="module")
def sample_cache(spark):
    df = spark.read.csv("tests/sample-input.txt", header=True, inferSchema=True)

    path = "tests/sample-output.sar"

    df.coalesce(1)\
        .write.format("eisber.sarplus")\
        .mode("overwrite")\
        .save(path)

    return path

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
def pandas_dummy_dataset(header):
    """Load sample dataset in pandas for testing; can be used to create a Spark dataframe
    Returns:
        single Pandas dataframe
    """
    ratings_dict = {
        header["col_user"]: [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
        header["col_item"]: [1, 2, 3, 4, 1, 2, 7, 8, 9, 10, 1, 2],
        header["col_rating"]: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
    return pd.DataFrame(ratings_dict)

@pytest.mark.spark
def test_good(spark, sample_cache):
    model = SARModel(sample_cache)
    y = model.predict([0, 1], [10, 20], top_k=10, remove_seen=False)

    assert_compare(0, 5, y[0])
    assert_compare(1, 44, y[1])
    assert_compare(2, 64, y[2])

@pytest.mark.spark
def test_good_less(spark, sample_cache):
    model = SARModel(sample_cache)
    y = model.predict([0, 2], [10, 3], top_k=5, remove_seen=False)

    assert_compare(0, 1, y[0])
    assert_compare(1, 11.6, y[1])
    assert_compare(2, 12.3, y[2])

@pytest.mark.spark
def test_good_require_sort(spark, sample_cache):
    model = SARModel(sample_cache)
    y = model.predict([1, 0], [20, 10], top_k=10, remove_seen=False)

    assert_compare(0, 5, y[0])
    assert_compare(1, 44, y[1])
    assert_compare(2, 64, y[2])

    assert 3 == len(y)

@pytest.mark.spark
def test_good_require_sort_remove_seen(spark, sample_cache):
    model = SARModel(sample_cache)
    y = model.predict([1, 0], [20, 10], top_k=10, remove_seen=True)

    assert_compare(2, 64, y[0])
    assert 1 == len(y)

@pytest.mark.spark
def test_pandas(spark, sample_cache):
    item_scores = pd.DataFrame([(0, 2.3), (1, 3.1)], columns=["itemID", "score"])

    model = SARModel(sample_cache)
    y = model.predict(item_scores["itemID"].values, item_scores["score"].values, top_k=10, remove_seen=False)

    assert_compare(0, 0.85, y[0])
    assert_compare(1, 6.9699, y[1])
    assert_compare(2, 9.92, y[2])

@pytest.mark.spark
def test_e2e(spark, pandas_dummy_dataset, header):
    sar = SARPlus(spark, **header)
    
    # TODO: cooccurence is broken
    df = spark.createDataFrame(pandas_dummy_dataset)
    sar.fit(df) 

    # assert 4*4 + 32 == sar.item_similarity.count()

    print(sar.item_similarity
        .toPandas()
        .pivot_table(index='i1', columns='i2', values='value'))

    test_df = spark.createDataFrame(pd.DataFrame({
        header['col_user']: [3],
        header['col_item']: [2]
    }))

    r1 = sar.recommend_k_items_slow(test_df, top_k=3)
    print("slow")
    print(r1.show())

    r2 = sar.recommend_k_items(test_df, "tests/test_e2e_cache", top_k=3, n_user_prediction_partitions=2, remove_seen=False)
    print("fast")
    print(r2.show())


