# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import pytest
from pandas.testing import assert_frame_equal

from pysarplus import SARPlus, SARModel


def assert_compare(expected_id, expected_score, actual_prediction):
    assert expected_id == actual_prediction.id
    assert math.isclose(
        expected_score, actual_prediction.score, rel_tol=1e-3, abs_tol=1e-3
    )


@pytest.fixture(scope="module")
def spark(tmp_path_factory, app_name="Sample", url="local[*]", memory="1G"):
    """Start Spark if not started
    Args:
        app_name (str): sets name of the application
        url (str): url for spark master
        memory (str): size of memory for spark driver
    """

    try:
        sarplus_jar_path = next(
            Path(__file__)
            .parents[2]
            .joinpath("scala", "target")
            .glob("**/sarplus*.jar")
        ).absolute()
    except StopIteration:
        raise Exception("Could not find Sarplus JAR file")

    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.jars", sarplus_jar_path)
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.default.parallelism", "1")
        .config("spark.sql.crossJoin.enabled", True)
        .config("spark.ui.enabled", False)
        .config("spark.sql.warehouse.dir", str(tmp_path_factory.mktemp("spark")))
        # .config("spark.eventLog.enabled", True) # only for local debugging, breaks on build server
        .getOrCreate()
    )

    return spark


@pytest.fixture(scope="module")
def sample_cache(spark):
    df = spark.read.csv("tests/sample-input.txt", header=True, inferSchema=True)

    path = "tests/sample-output.sar"

    df.coalesce(1).write.format("com.microsoft.sarplus").mode("overwrite").save(path)

    return path


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
    y = model.predict(
        item_scores["itemID"].values,
        item_scores["score"].values,
        top_k=10,
        remove_seen=False,
    )

    assert_compare(0, 0.85, y[0])
    assert_compare(1, 6.9699, y[1])
    assert_compare(2, 9.92, y[2])


@pytest.mark.spark
def test_e2e(spark, pandas_dummy_dataset, header):
    sar = SARPlus(spark, **header, cache_path="tests/test_e2e_cache")

    df = spark.createDataFrame(pandas_dummy_dataset)
    sar.fit(df)

    test_df = spark.createDataFrame(
        pd.DataFrame({header["col_user"]: [3], header["col_item"]: [2]})
    )

    r1 = (
        sar.recommend_k_items(test_df, top_k=3, remove_seen=False)
        .toPandas()
        .sort_values([header["col_user"], header["col_item"]])
        .reset_index(drop=True)
    )

    r2 = (
        sar.recommend_k_items(
            test_df,
            top_k=3,
            n_user_prediction_partitions=2,
            remove_seen=False,
            use_cache=True,
        )
        .toPandas()
        .sort_values([header["col_user"], header["col_item"]])
        .reset_index(drop=True)
    )

    assert (r1.iloc[:, :2] == r2.iloc[:, :2]).all().all()
    assert np.allclose(r1.score.values, r2.score.values, 1e-3)


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_fit(
    spark, similarity_type, timedecay_formula, train_test_dummy_timestamp, header
):
    model = SARPlus(
        spark,
        **header,
        timedecay_formula=timedecay_formula,
        similarity_type=similarity_type,
    )

    trainset, testset = train_test_dummy_timestamp

    df = spark.createDataFrame(trainset)
    df.write.mode("overwrite").saveAsTable("trainset")

    df = spark.table("trainset")

    model.fit(df)


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
    spark,
    threshold,
    similarity_type,
    file,
    demo_usage_data,
    sar_settings,
    header,
):

    model = SARPlus(
        spark,
        **header,
        timedecay_formula=False,
        time_decay_coefficient=30,
        time_now=None,
        threshold=threshold,
        similarity_type=similarity_type,
    )

    df = spark.createDataFrame(demo_usage_data)
    model.fit(df)

    # reference
    item_similarity_ref = pd.read_csv(
        sar_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    item_similarity_ref = pd.melt(
        item_similarity_ref,
        item_similarity_ref.columns[0],
        item_similarity_ref.columns[1:],
        "i2",
        "value",
    )
    item_similarity_ref.columns = ["i1", "i2", "value"]

    item_similarity_ref = (
        item_similarity_ref[item_similarity_ref.value > 0]
        .sort_values(["i1", "i2"])
        .reset_index(drop=True)
    )
    # actual
    item_similarity = (
        model.item_similarity.toPandas()
        .sort_values(["i1", "i2"])
        .reset_index(drop=True)
    )

    if similarity_type == "cooccurrence":
        assert (item_similarity_ref == item_similarity).all().all()
    else:
        assert (
            (item_similarity.iloc[:, :1] == item_similarity_ref.iloc[:, :1]).all().all()
        )

        assert np.allclose(
            item_similarity.value.values,
            item_similarity_ref.value.values,
            atol=sar_settings["ATOL"],
        )


# Test 7
def test_user_affinity(spark, demo_usage_data, sar_settings, header):
    time_now = demo_usage_data[header["col_timestamp"]].max()

    model = SARPlus(
        spark,
        **header,
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        similarity_type="cooccurrence",
    )

    df = spark.createDataFrame(demo_usage_data)
    model.fit(df)

    user_affinity_ref = pd.read_csv(sar_settings["FILE_DIR"] + "user_aff.csv")
    user_affinity_ref = pd.melt(
        user_affinity_ref,
        user_affinity_ref.columns[0],
        user_affinity_ref.columns[1:],
        "ItemId",
        "Rating",
    )
    user_affinity_ref = user_affinity_ref[user_affinity_ref.Rating > 0].reset_index(
        drop=True
    )

    # construct dataframe with test user id we'd like to get the affinity for
    df_test = spark.createDataFrame(
        pd.DataFrame({header["col_user"]: [sar_settings["TEST_USER_ID"]]})
    )
    user_affinity = model.get_user_affinity(df_test).toPandas().reset_index(drop=True)

    # verify the that item ids are the same
    assert (user_affinity[header["col_item"]] == user_affinity_ref.ItemId).all()

    assert np.allclose(
        user_affinity_ref[header["col_rating"]].values,
        user_affinity["Rating"].values,
        atol=sar_settings["ATOL"],
    )

    # Set time_now to 60 days later
    user_affinity_ref = (
        pd.read_csv(sar_settings["FILE_DIR"] + "user_aff_2_months_later.csv")
        .iloc[:, 1:]
        .squeeze()
    )
    user_affinity_ref = user_affinity_ref[user_affinity_ref > 0]

    two_months = 2 * 30 * (24 * 60 * 60)
    model = SARPlus(
        spark,
        **header,
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=demo_usage_data[header["col_timestamp"]].max() + two_months,
        similarity_type="cooccurrence",
    )
    model.fit(spark.createDataFrame(demo_usage_data))
    df_test = pd.DataFrame({header["col_user"]: [sar_settings["TEST_USER_ID"]]})
    df_test = spark.createDataFrame(df_test)
    user_affinity = model.get_user_affinity(df_test).toPandas()
    user_affinity = user_affinity.set_index(header["col_item"])[header["col_rating"]]
    user_affinity = user_affinity[user_affinity_ref.index]

    assert np.allclose(user_affinity_ref, user_affinity, atol=sar_settings["ATOL"])


# Tests 8-10
@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")],
)
def test_userpred(
    spark,
    tmp_path,
    threshold,
    similarity_type,
    file,
    header,
    sar_settings,
    demo_usage_data,
):
    time_now = demo_usage_data[header["col_timestamp"]].max()

    test_id = "{0}_{1}_{2}".format(threshold, similarity_type, file)

    model = SARPlus(
        spark,
        **header,
        table_prefix=test_id,
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        threshold=threshold,
        similarity_type=similarity_type,
        cache_path=str(tmp_path.joinpath("test_userpred-" + test_id)),
    )

    df = spark.createDataFrame(demo_usage_data)
    model.fit(df)

    url = sar_settings["FILE_DIR"] + "userpred_" + file + str(threshold) + "_userid_only.csv"

    pred_ref = pd.read_csv(url)
    pred_ref = (
        pd.wide_to_long(pred_ref, ["rec", "score"], "user", "idx")
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # Note: it's important to have a separate cache_path for each run as they're interfering with each other
    pred = model.recommend_k_items(
        spark.createDataFrame(
            demo_usage_data[
                demo_usage_data[header["col_user"]] == sar_settings["TEST_USER_ID"]
            ]
        ),
        top_k=10,
        n_user_prediction_partitions=1,
        use_cache=True,
    )

    pred = pred.toPandas().sort_values("score", ascending=False).reset_index(drop=True)

    assert (pred.MovieId.values == pred_ref.rec.values).all()
    assert np.allclose(
        pred.score.values, pred_ref.score.values, atol=sar_settings["ATOL"]
    )


@pytest.mark.spark
def test_get_popularity_based_topk(spark):
    # same df as in tests/unit/recommenders/models/test_sar_singlenode.py
    train_pd = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4],
            "item_id": [1, 4, 2, 1, 5, 4, 1, 4, 6, 3, 2, 4],
            "rating": [1, 2, 3, 1, 2, 3, 1, 2, 3, 3, 3, 1],
        }
    )
    train_df = spark.createDataFrame(train_pd)

    model = SARPlus(
        spark,
        col_user="user_id",
        col_item="item_id",
        col_rating="rating",
        col_timestamp="timestamp",
        similarity_type="jaccard",
    )
    model.fit(train_df)

    actual = model.get_popularity_based_topk(top_k=3).toPandas()
    expected = pd.DataFrame(
        {
            "item_id": [4, 1, 2],
            "frequency": [4, 3, 2],
        }
    )
    assert_frame_equal(expected, actual, check_dtype=False)


@pytest.mark.spark
def test_get_topk_most_similar_users(spark):
    # same df as in tests/unit/recommenders/models/test_sar_singlenode.py
    train_pd = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3, 3, 3, 3, 4, 4],
            "item_id": [1, 2, 1, 2, 3, 4, 5, 6, 1, 2],
            "rating": [3.0, 4.0, 3.0, 4.0, 3.0, 2.0, 1.0, 5.0, 5.0, 1.0],
        }
    )
    train_df = spark.createDataFrame(train_pd)

    model = SARPlus(
        spark,
        col_user="user_id",
        col_item="item_id",
        col_rating="rating",
        col_timestamp="timestamp",
        similarity_type="jaccard",
    )
    model.fit(train_df)

    actual = model.get_topk_most_similar_users(
        test=train_df, user=1, top_k=1
    ).toPandas()
    expected = pd.DataFrame(
        {
            "user_id": [2],
            "similarity": [25.0],
        }
    )
    assert_frame_equal(expected, actual, check_dtype=False)

    actual = model.get_topk_most_similar_users(
        test=train_df, user=2, top_k=1
    ).toPandas()
    expected = pd.DataFrame(
        {
            "user_id": [1],
            "similarity": [25.0],
        }
    )
    assert_frame_equal(expected, actual, check_dtype=False)

    actual = model.get_topk_most_similar_users(
        test=train_df, user=1, top_k=2
    ).toPandas()
    expected = pd.DataFrame(
        {
            "user_id": [2, 4],
            "similarity": [25.0, 19.0],
        }
    )
    assert_frame_equal(expected, actual, check_dtype=False)
