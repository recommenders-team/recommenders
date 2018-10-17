import sys
import os
import pytest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

from utilities.recommender.sar.sar_pyspark import SARpySparkReference
from utilities.recommender.sar import TIME_NOW
from utilities.common.constants import PREDICTION_COL
from tests.unit.test_sar_singlenode import _read_matrix, _load_userped, _load_affinity


@pytest.fixture(scope="module")
def get_train_test(load_pandas_dummy_timestamp_dataset):
    trainset, testset = train_test_split(
        load_pandas_dummy_timestamp_dataset, test_size=0.2, random_state=0
    )
    return trainset, testset


def _index_and_fit(spark, model, df_all, header):

    df_all.createOrReplaceTempView("df_all")

    # create new index for the items
    query = (
        "select "
        + header["col_user"]
        + ", "
        + "dense_rank() over(partition by 1 order by "
        + header["col_user"]
        + ") as row_id, "
        + header["col_item"]
        + ", "
        + "dense_rank() over(partition by 1 order by "
        + header["col_item"]
        + ") as col_id, "
        + header["col_rating"]
        + ", "
        + header["col_timestamp"]
        + " from df_all"
    )
    log.info("Running query -- " + query)
    df_all = spark.sql(query)
    df_all.createOrReplaceTempView("df_all")

    log.info("Obtaining all users and items ")
    # Obtain all the users and items from both training and test data
    unique_users = np.array(
        [
            x[header["col_user"]]
            for x in df_all.select(header["col_user"]).distinct().toLocalIterator()
        ]
    )
    unique_items = np.array(
        [
            x[header["col_item"]]
            for x in df_all.select(header["col_item"]).distinct().toLocalIterator()
        ]
    )

    log.info("Indexing users and items")
    # index all rows and columns, then split again intro train and test
    # We perform the reduction on Spark across keys before calling .collect so this is scalable
    index2user = dict(
        df_all.select(["row_id", header["col_user"]])
        .rdd.reduceByKey(lambda _, v: v)
        .collect()
    )
    index2item = dict(
        df_all.select(["col_id", header["col_item"]])
        .rdd.reduceByKey(lambda _, v: v)
        .collect()
    )

    # reverse the dictionaries: actual IDs to inner index
    user_map_dict = {v: k for k, v in index2user.items()}
    item_map_dict = {v: k for k, v in index2item.items()}

    # we need to index the train and test sets for SAR matrix operations to work
    model.set_index(
        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item
    )

    model.fit(df_all)

    return df_all


"""
Fixtures to load and reconcile custom output from TLC
"""


@pytest.fixture
def demo_usage_data_spark(spark, demo_usage_data, header):
    data_local = demo_usage_data[[x[1] for x in header.items()]]
    # TODO: install pyArrow in DS VM
    # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    data = spark.createDataFrame(data_local)
    return data


def rearrange_to_test_sql(array, row_ids, col_ids, row_map, col_map):
    """Rearranges SAR array into test array order
    Same as rearrange_to_test but offsets the count by -1 to account for SQL counts starting at 1"""
    if row_ids is not None:
        row_index = [row_map[x] - 1 for x in row_ids]
        array = array[row_index, :]
    if col_ids is not None:
        col_index = [col_map[x] - 1 for x in col_ids]
        array = array[:, col_index]
    return array


"""
Tests 
"""


@pytest.mark.spark
def test_initializaton_and_fit(header, spark, demo_usage_data_spark):
    """Test algorithm initialization"""

    # recommender will execute a fit method here
    model = SARpySparkReference(spark, **header)
    _index_and_fit(spark, model, demo_usage_data_spark, header)

    assert model is not None
    assert hasattr(model, "set_index")
    assert hasattr(model, "fit")
    assert hasattr(model, "recommend_k_items")


@pytest.mark.spark
def test_recommend_top_k(header, spark, demo_usage_data_spark):
    """Test algo recommend top-k"""

    # recommender will execute a fit method here
    model = SARpySparkReference(spark, **header)
    data_indexed = _index_and_fit(spark, model, demo_usage_data_spark, header)

    top_k_spark = model.recommend_k_items(data_indexed, top_k=10)
    top_k = top_k_spark.toPandas()

    assert 23410 == len(top_k)
    assert isinstance(top_k, pd.DataFrame)
    assert top_k[header["col_user"]].dtype == object
    assert top_k[header["col_item"]].dtype == object
    assert top_k[PREDICTION_COL].dtype == float


"""
Main SAR tests are below - load test files which are used for both Scala SAR and Python reference implementations
"""

# Tests 1-6
params = "threshold,similarity_type,file"


@pytest.mark.spark
@pytest.mark.parametrize(
    params,
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
    threshold,
    similarity_type,
    file,
    demo_usage_data_spark,
    header,
    spark,
    spark_test_settings,
):

    model = SARpySparkReference(
        spark, similarity_type=similarity_type, threshold=threshold, **header
    )
    _index_and_fit(spark, model, demo_usage_data_spark, header)

    true_item_similarity, row_ids, col_ids = _read_matrix(
        spark_test_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    test_item_similarity = rearrange_to_test_sql(
        model.get_item_similarity_as_matrix(),
        row_ids,
        col_ids,
        model.item_map_dict,
        model.item_map_dict,
    )

    if similarity_type is "cooccurrence":
        # these are integer counts so can test for direct equality
        assert np.array_equal(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
        )
    else:
        assert np.allclose(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
            atol=spark_test_settings["ATOL"],
        )


# Test 7
@pytest.mark.spark
def test_user_affinity(spark_test_settings, header, spark, demo_usage_data_spark):
    # time_now None should trigger max value computation from Data
    model = SARpySparkReference(
        spark,
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30,
        **header,
    )
    _index_and_fit(spark, model, demo_usage_data_spark, header)

    true_user_affinity, items = _load_affinity(
        spark_test_settings["FILE_DIR"] + "user_aff.csv"
    )

    tester_affinity = model.get_user_affinity_as_vector(
        spark_test_settings["TEST_USER_ID"]
    )

    test_user_affinity = np.reshape(
        rearrange_to_test_sql(tester_affinity, None, items, None, model.item_map_dict),
        -1,
    )
    assert np.allclose(
        true_user_affinity.astype(test_user_affinity.dtype),
        test_user_affinity,
        atol=spark_test_settings["ATOL"],
    )


# Tests 8-10
params = "threshold,similarity_type,file"


@pytest.mark.spark
@pytest.mark.parametrize(
    params, [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")]
)
def test_userpred(
    threshold,
    similarity_type,
    file,
    header,
    spark,
    demo_usage_data_spark,
    spark_test_settings,
):

    # time_now None should trigger max value computation from Data
    model = SARpySparkReference(
        spark,
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30,
        threshold=threshold,
        **header,
    )
    data_indexed = _index_and_fit(spark, model, demo_usage_data_spark, header)

    true_items, true_scores = _load_userped(
        spark_test_settings["FILE_DIR"]
        + "userpred_"
        + file
        + str(threshold)
        + "_userid_only.csv"
    )

    data_indexed.createOrReplaceTempView("data_indexed")
    test_data = spark.sql(
        "select * from data_indexed where row_id = %d"
        % model.user_map_dict[spark_test_settings["TEST_USER_ID"]]
    )
    test_results = model.recommend_k_items(test_data, top_k=10).toPandas()
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=spark_test_settings["ATOL"])

