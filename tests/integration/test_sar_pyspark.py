import pytest

from airship.evaluation.evaluator import RankingEvaluator
from airship.recommenders.tests.test_sar_pyspark import setup_SARpySpark
from airship.tests.spark_fixtures import ml100k, ml1m
from airship.tests.data_fixtures import header

@pytest.mark.parametrize("dataset,ndcg", [
    (ml100k, 0.0),
    (ml1m, 0.0)
])
@pytest.mark.integration
def test_int_sar_sql(dataset, ndcg):
    from airship.tests.spark_fixtures import start_spark_test

    spark = start_spark_test()
    data = dataset()

    # index and fit the data
    recommender = setup_SARpySpark(spark, data, timedecay_formula = False, **header)

    top_k = recommender.model.recommend_k_items(recommender.data_indexed)
    assert top_k.count() > 0

    # we are not using timestamp in the evaluation methods
    del header["col_timestamp"]
    ranking_evaluator = RankingEvaluator(data, top_k, relevancy_method="top_k", spark=spark, **header)

    assert pytest.approx(ndcg, 0.1) == ranking_evaluator.ndcg_at_k()

