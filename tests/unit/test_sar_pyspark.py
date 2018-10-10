import sys
import os
import pytest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import logging


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# TODO: better solution??
root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
sys.path.append(root)

from utilities.recommender.sar.sar_pyspark import SARpySparkReference
from utilities.recommender.sar import TIME_NOW
from tests.conftest import header, start_spark_test
from utilities.common.constants import PREDICTION_COL
from tests.unit.test_sar_singlenode import load_demoUsage_data, read_matrix, load_userped, load_affinity

# absolute tolerance parameter for matrix equivalnce in SAR tests
ATOL = 1e-8
# directory of the current file - used to link unit test data
FILE_DIR = "http://recodatasets.blob.core.windows.net/sarunittest/"
# user ID used in the test files (they are designed for this user ID, this is part of the test)
TEST_USER_ID = "0003000098E85347"

# convenient way to invoke SAR with different parameters on different datasets
# TODO: pytest class fixtures are not yet supported as of this release
# @pytest.fixture
class setup_SARpySpark:
    def __init__(
        self,
        spark,
        data,
        remove_seen=True,
        similarity_type="jaccard",
        timedecay_formula=False,
        time_decay_coefficient=30,
        threshold=1,
        time_now=TIME_NOW,
    ):

        self.data = data
        model = SARpySparkReference(
            spark,
            remove_seen=remove_seen,
            similarity_type=similarity_type,
            timedecay_formula=timedecay_formula,
            time_decay_coefficient=time_decay_coefficient,
            time_now=time_now,
            threshold=threshold,
            **header()
        )

        data_indexed, unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = airship_hash_sql(
            spark, data
        )

        # we need to index the train and test sets for SAR matrix operations to work
        model.set_index(
            unique_users,
            unique_items,
            user_map_dict,
            item_map_dict,
            index2user,
            index2item,
        )
        model.fit(data_indexed)

        self.model = model
        self.data_indexed = data_indexed
        self.user_map_dict = user_map_dict
        self.item_map_dict = item_map_dict


@pytest.fixture(scope="module")
def get_train_test(load_pandas_dummy_timestamp_dataset):
    trainset, testset = train_test_split(
        load_pandas_dummy_timestamp_dataset, test_size=0.2, random_state=0
    )
    return trainset, testset


def airship_hash_sql(spark, df_all):

    df_all.createOrReplaceTempView("df_all")

    # create new index for the items
    query = (
        "select "
        + header()["col_user"]
        + ", "
        + "dense_rank() over(partition by 1 order by "
        + header()["col_user"]
        + ") as row_id, "
        + header()["col_item"]
        + ", "
        + "dense_rank() over(partition by 1 order by "
        + header()["col_item"]
        + ") as col_id, "
        + header()["col_rating"]
        + ", "
        + header()["col_timestamp"]
        + " from df_all"
    )
    log.info("Running query -- " + query)
    df_all = spark.sql(query)
    df_all.createOrReplaceTempView("df_all")

    log.info("Obtaining all users and items ")
    # Obtain all the users and items from both training and test data
    unique_users = np.array(
        [
            x[header()["col_user"]]
            for x in df_all.select(header()["col_user"]).distinct().toLocalIterator()
        ]
    )
    unique_items = np.array(
        [
            x[header()["col_item"]]
            for x in df_all.select(header()["col_item"]).distinct().toLocalIterator()
        ]
    )

    log.info("Indexing users and items")
    # index all rows and columns, then split again intro train and test
    # We perform the reduction on Spark across keys before calling .collect so this is scalable
    index2user = dict(
        df_all.select(["row_id", header()["col_user"]])
        .rdd.reduceByKey(lambda _, v: v)
        .collect()
    )
    index2item = dict(
        df_all.select(["col_id", header()["col_item"]])
        .rdd.reduceByKey(lambda _, v: v)
        .collect()
    )

    # reverse the dictionaries: actual IDs to inner index
    user_map_dict = {v: k for k, v in index2user.items()}
    item_map_dict = {v: k for k, v in index2item.items()}

    return (
        df_all,
        unique_users,
        unique_items,
        user_map_dict,
        item_map_dict,
        index2user,
        index2item,
    )


"""
Fixtures to load and reconcile custom output from TLC
"""

@pytest.fixture
def load_demoUsage_data_spark(spark):
    data_local = load_demoUsage_data()[[x[1] for x in header().items()]]
    # TODO: install pyArrow in DS VM
    # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    data = spark.createDataFrame(data_local)
    return data

def rearrange_to_test_sql(array, row_ids, col_ids, row_map, col_map):
    """Rearranges SAR array into test array order
    Same as rearrange_to_test but offsets the count by -1 to account for SQL counts starting at 1"""
    if row_ids is not None:
        row_index = [row_map[x]-1 for x in row_ids]
        array = array[row_index, :]
    if col_ids is not None:
        col_index = [col_map[x]-1 for x in col_ids]
        array = array[:, col_index]
    return array


"""
Tests 
"""
def test_initializaton_and_fit():
    """Test algorithm initialization"""

    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)
    # recommender will execute a fit method here
    recommender = setup_SARpySpark(spark, data)

    assert recommender is not None
    assert hasattr(recommender.model, 'set_index')
    assert hasattr(recommender.model, 'fit')
    assert hasattr(recommender.model, 'recommend_k_items')

def test_recommend_top_k():
    """Test algo recommend top-k"""

    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)

    # recommender will execute a fit method here
    recommender = setup_SARpySpark(spark, data)

    top_k_spark = recommender.model.recommend_k_items(recommender.data_indexed, top_k=10)
    top_k = top_k_spark.toPandas()

    assert 23410 == len(top_k)
    assert isinstance(top_k, pd.DataFrame)
    assert top_k[header()['col_user']].dtype == object
    assert top_k[header()['col_item']].dtype == object
    assert top_k[PREDICTION_COL].dtype == float

"""
Main SAR tests are below - load test files which are used for both Scala SAR and Python reference implementations
"""

# Tests 1-6
params="threshold,similarity_type,file"
@pytest.mark.parametrize(params, [
    (1,'cooccurrence', 'count'),
    (1,'jaccard', 'jac'),
    (1,'lift', 'lift'),
    (3,'cooccurrence', 'count'),
    (3,'jaccard', 'jac'),
    (3,'lift', 'lift')
])
def test_sar_item_similarity(threshold, similarity_type, file):
    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)
    tester = setup_SARpySpark(spark, data, similarity_type=similarity_type, threshold=threshold)
    true_item_similarity, row_ids, col_ids = read_matrix(FILE_DIR + 'sim_' + file + str(threshold) + '.csv')

    test_item_similarity = rearrange_to_test_sql(tester.model.get_item_similarity_as_matrix(), row_ids, col_ids,
                                                 tester.item_map_dict, tester.item_map_dict)
    if similarity_type is "cooccurrence":
        # these are integer counts so can test for direct equality
        assert np.array_equal(true_item_similarity.astype(test_item_similarity.dtype), test_item_similarity)
    else:
        assert np.allclose(true_item_similarity.astype(test_item_similarity.dtype), test_item_similarity, atol=ATOL)

# Test 7
def test_user_affinity():
    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)

    # time_now None should trigger max value computation from Data
    tester = setup_SARpySpark(spark, data, similarity_type='cooccurrence', timedecay_formula=True, time_now=None,
       time_decay_coefficient = 30)

    true_user_affinity, items = load_affinity(FILE_DIR+'user_aff.csv')

    tester_affinity = tester.model.get_user_affinity_as_vector(TEST_USER_ID)

    test_user_affinity = np.reshape(
        rearrange_to_test_sql(tester_affinity, None, items, None, tester.item_map_dict), -1)
    assert np.allclose(true_user_affinity.astype(test_user_affinity.dtype), test_user_affinity, atol=ATOL)

# Tests 8-10
params="threshold,similarity_type,file"
@pytest.mark.parametrize(params, [
    (3,'cooccurrence', 'count'),
    (3,'jaccard', 'jac'),
    (3,'lift', 'lift')
])
def test_userpred(threshold, similarity_type, file):
    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)

    # time_now None should trigger max value computation from Data
    tester = setup_SARpySpark(spark, data, remove_seen=True, similarity_type=similarity_type, timedecay_formula=True,
                       time_now=None, time_decay_coefficient=30, threshold=threshold)
    true_items, true_scores = load_userped(FILE_DIR + "userpred_" + file + str(threshold) + "_userid_only.csv")

    tester.data_indexed.createOrReplaceTempView("data_indexed")
    test_data = \
        spark.sql("select * from data_indexed where row_id = %d" % tester.user_map_dict[TEST_USER_ID])
    test_results = tester.model.recommend_k_items(test_data, top_k=10).toPandas()
    test_items = list(test_results[header()["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=ATOL)



