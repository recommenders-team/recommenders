import sys
import os
import pytest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from utilities.recommender.sar.sar_pyspark_sql import SARpySparkSQLReference
from utilities.recommender.sar import TIME_NOW
from utilities.common.constants import PREDICTION_COL
from tests.unit.test_sar_singlenode import _read_matrix, _load_userped, _load_affinity


@pytest.fixture(scope="module")
def get_train_test(load_pandas_dummy_timestamp_dataset):
    trainset, testset = train_test_split(
        load_pandas_dummy_timestamp_dataset, test_size=0.2, random_state=0
    )
    return trainset, testset

@pytest.fixture
def demo_usage_data_spark(spark, demo_usage_data, header):
    data_local = demo_usage_data[[x[1] for x in header.items()]]
    # TODO: install pyArrow in DS VM
    # spark.conf.set("spark.sql.execution.arrow.enabled", "true")
    data = spark.createDataFrame(data_local)
    return data


@pytest.mark.spark
def test_initializaton_and_fit(header, spark, demo_usage_data_spark):
    """Test algorithm initialization"""

    # recommender will execute a fit method here
    model = SARpySparkSQLReference(spark, **header)
    
    model.fit(demo_usage_data_spark)

    assert model is not None
    assert hasattr(model, "fit")
    assert hasattr(model, "recommend_k_items")

# Tests 1-6
@pytest.mark.spark
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
    threshold,
    similarity_type,
    file,
    demo_usage_data_spark,
    header,
    spark,
    sar_test_settings,
):

    model = SARpySparkSQLReference(spark, **header)

    model.fit(demo_usage_data_spark, similarity_type=similarity_type, threshold=threshold)

    true_item_similarity, row_ids, col_ids = _read_matrix(
        sar_test_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    # TODO...
    # test_item_similarity = rearrange_to_test_sql(
        # model.get_item_similarity_as_matrix(),
        # row_ids,
        # col_ids,
        # model.item_map_dict,
        # model.item_map_dict,
    # )

  # if similarity_type is "cooccurrence":
  #     # these are integer counts so can test for direct equality
  #     assert np.array_equal(
  #         true_item_similarity.astype(test_item_similarity.dtype),
  #         test_item_similarity,
  #     )
  # else:
  #     assert np.allclose(
  #         true_item_similarity.astype(test_item_similarity.dtype),
  #         test_item_similarity,
  #         atol=sar_test_settings["ATOL"],
  #     )


# Test 7
@pytest.mark.spark
def test_user_affinity(sar_test_settings, header, spark, demo_usage_data_spark):
    # time_now None should trigger max value computation from Data
    model = SARpySparkSQLReference(spark, **header)

    model.fit(
        demo_usage_data_spark, 
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30)

    true_user_affinity, items = _load_affinity(
        sar_test_settings["FILE_DIR"] + "user_aff.csv"
    )

    # TODO: ???
    tester_affinity = model.get_user_affinity(sar_test_settings["TEST_USER_ID"])

    print(tester_affinity)

    print(true_user_affinity)
   #tester_affinity = model.get_user_affinity_as_vector(
   #    sar_test_settings["TEST_USER_ID"]
   #)
#
   #test_user_affinity = np.reshape(
   #    rearrange_to_test_sql(tester_affinity, None, items, None, model.item_map_dict),
   #    -1,
   #)
   #assert np.allclose(
   #    true_user_affinity.astype(test_user_affinity.dtype),
   #    test_user_affinity,
   #    atol=sar_test_settings["ATOL"],
   #)

