import pytest

from airship.recommenders.tests.test_sar_pyspark import setup_SARpySpark
from airship.evaluation.evaluator import RankingEvaluator
from airship.tests.data_fixtures import header
from airship.recommenders.tests.test_sar_ref import sar_algo
from airship.tests.data_fixtures import load_pandas_dummy_dataset, \
    load_pandas_dummy_timestamp_dataset
from airship.tests.splitter_fixtures import random_splitter
from airship.tests.test_common import sar_ref_algo


# 1 basic SAR on notimedecay dataset
# 2 timedecay SAR on timedecay dataset
@pytest.mark.parametrize("splitter,model,dataset,ndcg", [
    (random_splitter, sar_algo()[0], load_pandas_dummy_dataset, 0.8293121225775372),
    (random_splitter, sar_algo()[1], load_pandas_dummy_timestamp_dataset, 0.8293121225775372)
])
@pytest.mark.smoke
def test_smoke_sar(splitter, model, dataset, ndcg):
    sar_ref_algo(splitter(), model[0], dataset(), ndcg=ndcg)

# pySpark reference implementation
@pytest.mark.parametrize("ndcg", [
    (0.0)
])
@pytest.mark.smoke
@pytest.mark.spark
def test_smoke_sar_pyspark(ndcg):

    from airship.tests.spark_fixtures import dummy, start_spark_test

    spark = start_spark_test()
    data = dummy(timestamp=True)

    # index and fit the data
    recommender = setup_SARpySpark(spark, data, **header)

    top_k = recommender.model.recommend_k_items(recommender.data_indexed)
    assert top_k.count() > 0

    # we are not using timestamp in the evaluation methods
    del header["col_timestamp"]
    ranking_evaluator = RankingEvaluator(data, top_k, relevancy_method="top_k", spark=spark, **header)

    assert pytest.approx(ndcg, 0.1) == ranking_evaluator.ndcg_at_k()


