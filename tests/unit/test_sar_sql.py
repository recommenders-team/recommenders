# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import numpy as np
import pandas as pd
try:
    from reco_utils.recommender.sar.sar_sql import SARSQLReference
except ModuleNotFoundError:
    pass  # skip this import if we are in pure python environment

from reco_utils.common.constants import PREDICTION_COL
from tests.sar_common import (
    read_matrix,
    load_userpred,
    load_affinity,
    rearrange_to_test_sql,
    index_and_fit_sql,
)


@pytest.mark.spark
def test_initializaton_and_fit(header, spark, demo_usage_data_spark):
    """Test algorithm initialization"""

    model = SARSQLReference(spark, **header)
    index_and_fit_sql(spark, model, demo_usage_data_spark, header)

    assert model is not None
    assert hasattr(model, "set_index")
    assert hasattr(model, "fit")
    assert hasattr(model, "recommend_k_items")


@pytest.mark.spark
def test_recommend_top_k(header, spark, demo_usage_data_spark):
    """Test algo recommend top-k"""

    # recommender will execute a fit method here
    model = SARSQLReference(spark, **header)
    data_indexed = index_and_fit_sql(spark, model, demo_usage_data_spark, header)

    top_k_spark = model.recommend_k_items(data_indexed, top_k=10)
    top_k = top_k_spark.toPandas()

    # in SQL implementation we output zero scores for some users and items, so the user-item count is different to the
    # pure pySpark SQL implementation
    assert 23429 == len(top_k)
    assert isinstance(top_k, pd.DataFrame)
    assert top_k[header["col_user"]].dtype == object
    assert top_k[header["col_item"]].dtype == object
    assert top_k[PREDICTION_COL].dtype == float


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
    threshold, similarity_type, file, demo_usage_data_spark, header, spark, sar_settings
):

    model = SARSQLReference(
        spark,
        similarity_type=similarity_type,
        threshold=threshold,
        **header
    )
    index_and_fit_sql(spark, model, demo_usage_data_spark, header)

    true_item_similarity, row_ids, col_ids = read_matrix(
        sar_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
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
            atol=sar_settings["ATOL"],
        )


# Test 7
@pytest.mark.spark
def test_user_affinity(sar_settings, header, spark, demo_usage_data_spark):
    # time_now None should trigger max value computation from Data
    model = SARSQLReference(
        spark,
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30,
        **header
    )
    index_and_fit_sql(spark, model, demo_usage_data_spark, header)

    true_user_affinity, items = load_affinity(sar_settings["FILE_DIR"] + "user_aff.csv")

    tester_affinity = model.get_user_affinity_as_vector(sar_settings["TEST_USER_ID"])

    test_user_affinity = np.reshape(
        rearrange_to_test_sql(tester_affinity, None, items, None, model.item_map_dict),
        -1,
    )
    assert np.allclose(
        true_user_affinity.astype(test_user_affinity.dtype),
        test_user_affinity,
        atol=sar_settings["ATOL"],
    )


# Tests 8-10
@pytest.mark.spark
@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")],
)
def test_userpred(
    threshold, similarity_type, file, header, spark, demo_usage_data_spark, sar_settings
):
    # time_now None should trigger max value computation from Data
    model = SARSQLReference(
        spark,
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30,
        threshold=threshold,
        **header
    )
    data_indexed = index_and_fit_sql(spark, model, demo_usage_data_spark, header)

    true_items, true_scores = load_userpred(
        sar_settings["FILE_DIR"]
        + "userpred_"
        + file
        + str(threshold)
        + "_userid_only.csv"
    )

    data_indexed.createOrReplaceTempView("data_indexed")
    test_data = spark.sql(
        "select * from data_indexed where row_id = %d"
        % model.user_map_dict[sar_settings["TEST_USER_ID"]]
    )
    test_results = model.recommend_k_items(test_data, top_k=10).toPandas()
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=sar_settings["ATOL"])
