import calendar
import datetime
import math
import numpy as np
import pandas as pd
import pytest
import os
from sklearn.model_selection import train_test_split

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
        .config("spark.jars", os.path.dirname(__file__) + "/../../scala/target/scala-2.11/sarplus_2.11-0.2.6.jar")
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.crossJoin.enabled", True)
        .config("spark.ui.enabled", False)
        # .config("spark.eventLog.enabled", True) # only for local debugging, breaks on build server
        .getOrCreate()
    )

    return spark

@pytest.fixture(scope="module")
def sample_cache(spark):
    df = spark.read.csv("tests/sample-input.txt", header=True, inferSchema=True)

    path = "tests/sample-output.sar"

    df.coalesce(1)\
        .write.format("com.microsoft.sarplus")\
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
    
    df = spark.createDataFrame(pandas_dummy_dataset)
    sar.fit(df) 

    # assert 4*4 + 32 == sar.item_similarity.count()

    # print(sar.item_similarity
        # .toPandas()
        # .pivot_table(index='i1', columns='i2', values='value'))

    test_df = spark.createDataFrame(pd.DataFrame({
        header['col_user']: [3],
        header['col_item']: [2]
    }))

    r1 = sar.recommend_k_items_slow(test_df, top_k=3, remove_seen=False)\
        .toPandas()\
        .sort_values([header['col_user'], header['col_item']])\
        .reset_index(drop=True)

    r2 = sar.recommend_k_items(test_df, "tests/test_e2e_cache", top_k=3, n_user_prediction_partitions=2, remove_seen=False)\
        .toPandas()\
        .sort_values([header['col_user'], header['col_item']])\
        .reset_index(drop=True)

    assert (r1.iloc[:,:2] == r2.iloc[:,:2]).all().all()
    assert np.allclose(
        r1.score.values,
        r2.score.values,
        1e-3
    )

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
    # TODO: install pyArrow in DS VM
    # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    data = spark.createDataFrame(data_local)
    return data


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


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_fit(spark, similarity_type, timedecay_formula, train_test_dummy_timestamp, header):
    model = SARPlus(spark, **header)
    
    trainset, testset = train_test_dummy_timestamp

    df = spark.createDataFrame(trainset)
    df.write.mode("overwrite").saveAsTable("trainset")

    df = spark.table("trainset")

    model.fit(df, 
       timedecay_formula=timedecay_formula,
       similarity_type=similarity_type)


"""
Main SAR tests are below - load test files which are used for both Scala SAR and Python reference implementations
"""

# Tests 1-6
@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [
        (1, "cooccurrence", "count"),
        (1, "jaccard", "jac"),
        (1, "lift", "lift"),
        (3, "cooccurrence", "count"),
        (3, "jaccard", "jac"),
        (3, "lift", "lift"),
    ],
)
def test_sar_item_similarity(
    spark, threshold, similarity_type, file, demo_usage_data, sar_settings, header
):

    model = SARPlus(spark, **header)

    df = spark.createDataFrame(demo_usage_data)
    model.fit(df, 
       timedecay_formula=False,
       time_decay_coefficient=30,
       time_now=None,
       threshold=threshold,
       similarity_type=similarity_type)

    # reference
    item_similarity_ref = pd.read_csv(sar_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv")

    item_similarity_ref = pd.melt(item_similarity_ref,
        item_similarity_ref.columns[0],
        item_similarity_ref.columns[1:],
        'i2',
        'value')
    item_similarity_ref.columns = ['i1', 'i2', 'value']

    item_similarity_ref = item_similarity_ref[item_similarity_ref.value > 0]\
        .sort_values(['i1', 'i2'])\
        .reset_index(drop=True)\

    # actual
    item_similarity = model.item_similarity\
        .toPandas()\
        .sort_values(['i1', 'i2'])\
        .reset_index(drop=True)

    if similarity_type is "cooccurrence":
        assert((item_similarity_ref == item_similarity).all().all())
    else:
        assert((item_similarity.iloc[:,:1] == item_similarity_ref.iloc[:,:1]).all().all())

        assert np.allclose(
            item_similarity.value.values,
            item_similarity_ref.value.values
        )

# Test 7
def test_user_affinity(spark, demo_usage_data, sar_settings, header):
    time_now = demo_usage_data[header["col_timestamp"]].max()

    model = SARPlus(spark, **header)

    df = spark.createDataFrame(demo_usage_data)
    model.fit(df, 
       timedecay_formula=True,
       time_decay_coefficient=30,
       time_now=time_now,
       similarity_type="cooccurrence")

    user_affinity_ref = pd.read_csv(sar_settings["FILE_DIR"] + "user_aff.csv")
    user_affinity_ref = pd.melt(user_affinity_ref, user_affinity_ref.columns[0], user_affinity_ref.columns[1:], 'ItemId', 'Rating')
    user_affinity_ref = user_affinity_ref[user_affinity_ref.Rating > 0]\
        .reset_index(drop=True)

    # construct dataframe with test user id we'd like to get the affinity for
    df_test = spark.createDataFrame(pd.DataFrame({header['col_user']:[sar_settings["TEST_USER_ID"]]}))
    user_affinity = model.get_user_affinity(df_test).toPandas().reset_index(drop=True)

    # verify the that item ids are the same
    assert (user_affinity[header['col_item']] == user_affinity_ref.ItemId).all()

    assert np.allclose(
        user_affinity_ref[header['col_rating']].values,
        user_affinity['Rating'].values,
        atol=sar_settings["ATOL"]
    )


# Tests 8-10
@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")],
)
def test_userpred(
    spark, threshold, similarity_type, file, header, sar_settings, demo_usage_data
):
    time_now = demo_usage_data[header["col_timestamp"]].max()

    test_id = '{0}_{1}_{2}'.format(threshold, similarity_type, file)

    model = SARPlus(spark, **header, table_prefix=test_id)

    df = spark.createDataFrame(demo_usage_data)
    model.fit(df, 
       timedecay_formula=True,
       time_decay_coefficient=30,
       time_now=time_now,
       threshold=threshold,
       similarity_type=similarity_type)

    url = (sar_settings["FILE_DIR"]
        + "userpred_"
        + file
        + str(threshold)
        + "_userid_only.csv")

    pred_ref = pd.read_csv(url)
    pred_ref = pd.wide_to_long(pred_ref, ['rec','score'], 'user', 'idx')\
        .sort_values('score', ascending=False)\
        .reset_index(drop=True)

    # Note: it's important to have a separate cache_path for each run as they're interferring with each other
    pred = model.recommend_k_items(
        spark.createDataFrame(demo_usage_data[
           demo_usage_data[header["col_user"]] == sar_settings["TEST_USER_ID"]
        ]),
        cache_path='test_userpred-' + test_id,
        top_k=10,
        n_user_prediction_partitions=1)

    pred = pred.toPandas()\
        .sort_values('score', ascending=False)\
        .reset_index(drop=True)

    assert (pred.MovieId.values == pred_ref.rec.values).all()
    assert np.allclose(pred.score.values, pred_ref.score.values, atol=sar_settings["ATOL"])