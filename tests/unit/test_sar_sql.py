import pytest
import numpy as np
import pandas as pd
from reco_utils.recommender.sar.sar_sql import SARSQLReference
from reco_utils.common.constants import PREDICTION_COL
from tests.unit.sar_common import (
    read_matrix,
    load_userpred,
    load_affinity,
    _rearrange_to_test_sql,
    _index_and_fit_sql,
)
from tests.unit.sar_common import demo_usage_data_spark

# convenient way to invoke SAR with different parameters on different datasets
# TODO: pytest class fixtures are not yet supported as of this release
# @pytest.fixture
class setup_SARSQL:
    def __init__(
        self,
        spark,
        data,
        remove_seen=True,
        similarity_type="jaccard",
        timedecay_formula=False,
        time_decay_coefficient=30,
        time_now=None,
        **header,
    ):

        self.data = data
        model = SARSQLReference(
            spark,
            remove_seen=remove_seen,
            similarity_type=similarity_type,
            timedecay_formula=timedecay_formula,
            time_decay_coefficient=time_decay_coefficient,
            time_now=time_now,
            **header,
        )

        data_indexed, unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = _index_and_fit_sql(
            spark, data, header
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


@pytest.mark.spark
def test_initializaton_and_fit(header, spark, demo_usage_data_spark):
    """Test algorithm initialization"""

    # recommender will execute a fit method here
    model = SARSQLReference(spark, demo_usage_data_spark, **header)
    # test running indexer
    _index_and_fit_sql(spark, demo_usage_data_spark, header)

    assert model is not None
    assert hasattr(model, "set_index")
    assert hasattr(model, "fit")
    assert hasattr(model, "recommend_k_items")


@pytest.mark.spark
def test_recommend_top_k(header, spark, demo_usage_data_spark):
    """Test algo recommend top-k"""

    # recommender will execute a fit method here
    algo = setup_SARSQL(spark, demo_usage_data_spark, **header)
    model = algo.model

    top_k_spark = model.recommend_k_items(algo.data_indexed, top_k=10)
    top_k = top_k_spark.toPandas()

    # in SQL implementation we output zero scores for some users and items, so the user-item count is different to the
    # pure pySpark SQL implementation
    assert 23429 == len(top_k)
    assert isinstance(top_k, pd.DataFrame)
    assert top_k[header["col_user"]].dtype == object
    assert top_k[header["col_item"]].dtype == object
    assert top_k[PREDICTION_COL].dtype == float
    # TODO: add validation of the topk result


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
    algo = setup_SARSQL(
        spark,
        demo_usage_data_spark,
        similarity_type=similarity_type,
        threshold=threshold,
        **header,
    )
    model = algo.model

    true_item_similarity, row_ids, col_ids = read_matrix(
        sar_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    test_item_similarity = _rearrange_to_test_sql(
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
    algo = setup_SARSQL(
        spark,
        demo_usage_data_spark,
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30,
        **header,
    )
    model = algo.model

    true_user_affinity, items = load_affinity(sar_settings["FILE_DIR"] + "user_aff.csv")

    tester_affinity = model.get_user_affinity_as_vector(sar_settings["TEST_USER_ID"])

    test_user_affinity = np.reshape(
        _rearrange_to_test_sql(tester_affinity, None, items, None, model.item_map_dict),
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
    algo = setup_SARSQL(
        spark,
        demo_usage_data_spark,
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=True,
        time_now=None,
        time_decay_coefficient=30,
        threshold=threshold,
        **header,
    )
    model = algo.model

    true_items, true_scores = load_userpred(
        sar_settings["FILE_DIR"]
        + "userpred_"
        + file
        + str(threshold)
        + "_userid_only.csv"
    )

    algo.data_indexed.createOrReplaceTempView("data_indexed")
    test_data = spark.sql(
        "select * from data_indexed where row_id = %d"
        % model.user_map_dict[sar_settings["TEST_USER_ID"]]
    )
    test_results = model.recommend_k_items(test_data, top_k=10).toPandas()
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=sar_settings["ATOL"])
