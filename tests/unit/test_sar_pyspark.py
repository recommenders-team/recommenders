"""
====================================================================
SAR Family Tests
====================================================================
"""
import pandas as pd
import pytest
import itertools
import os
import time
import datetime
import csv
import numpy as np
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

from airship.tests.spark_fixtures import start_spark_test, dummy

from airship.recommenders.microsoft.sar.sar_ref_pyspark import SARpySparkReference
from airship.recommenders.microsoft.sar.sar_family import TIME_NOW
from airship.recommenders.recommender_base import PREDICTION_COL
from airship.tests.data_fixtures import header
from airship.recommenders.tests.test_sar_ref import read_matrix, load_affinity, load_userped
from airship.recommenders.tests.test_sar_sql import rearrange_to_test_sql, load_demoUsage_data_spark, airship_hash_sql

# absolute tolerance parameter for matrix equivalnce in SAR tests
ATOL=1e-8
# directory of the current file - used to link unit test data
FILE_DIR=os.path.dirname(os.path.abspath(__file__))
# user ID used in the test files (they are designed for this user ID, this is part of the test)
TEST_USER_ID='0003000098E85347'

# convenient way to invoke SAR with different parameters on different datasets
# TODO: pytest class fixtures are not yet supported as of this release
# @pytest.fixture
class setup_SARpySpark:
    def __init__(self, spark, data, remove_seen=True, similarity_type='jaccard',
                         timedecay_formula=False, time_decay_coefficient=30,
                         time_now=TIME_NOW, **header):

        self.data = data
        model = SARpySparkReference(spark, remove_seen=remove_seen, similarity_type=similarity_type,
                                  timedecay_formula=timedecay_formula, time_decay_coefficient=time_decay_coefficient,
                                  time_now=time_now, **header)

        data_indexed, unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item =\
            airship_hash_sql(spark, data)

        # we need to index the train and test sets for SAR matrix operations to work
        model.set_index(unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item)
        model.fit(data_indexed)

        self.model = model
        self.data_indexed = data_indexed
        self.user_map_dict = user_map_dict
        self.item_map_dict = item_map_dict

"""
Tests 
"""
def test_initializaton_and_fit():
    """Test algorithm initialization"""

    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)
    # recommender will execute a fit method here
    recommender = setup_SARpySpark(spark, data, **header)

    assert recommender is not None
    assert hasattr(recommender.model, 'set_index')
    assert hasattr(recommender.model, 'fit')
    assert hasattr(recommender.model, 'recommend_k_items')

def test_recommend_top_k():
    """Test algo recommend top-k"""

    spark = start_spark_test()
    data = load_demoUsage_data_spark(spark)

    # recommender will execute a fit method here
    recommender = setup_SARpySpark(spark, data, **header)

    top_k_spark = recommender.model.recommend_k_items(recommender.data_indexed, top_k=10)
    top_k = top_k_spark.toPandas()

    assert 23410 == len(top_k)
    assert isinstance(top_k, pd.DataFrame)
    assert top_k[header['col_user']].dtype == object
    assert top_k[header['col_item']].dtype == object
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
    tester = setup_SARpySpark(spark, data, similarity_type=similarity_type, threshold=threshold, **header)
    true_item_similarity, row_ids, col_ids = read_matrix(FILE_DIR + '/data/sim_' + file + str(threshold) + '.csv')

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
       time_decay_coefficient = 30, **header)

    true_user_affinity, items = load_affinity(FILE_DIR+'/data/user_aff.csv')

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
                       time_now=None, time_decay_coefficient=30, threshold=threshold, **header)
    true_items, true_scores = load_userped(FILE_DIR + "/data/userpred_" + file + str(threshold) + "_userid_only.csv")

    tester.data_indexed.createOrReplaceTempView("data_indexed")
    test_data = \
        spark.sql("select * from data_indexed where row_id = %d" % tester.user_map_dict[TEST_USER_ID])
    test_results = tester.model.recommend_k_items(test_data, top_k=10).toPandas()
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=ATOL)



