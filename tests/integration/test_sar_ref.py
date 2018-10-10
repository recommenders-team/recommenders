import pytest

from airship.recommenders.tests.test_sar_ref import sar_algo
from airship.tests.splitter_fixtures import random_splitter
from airship.tests.test_common import sar_ref_algo


@pytest.fixture
def ml100k_data():
    from airship.tests.spark_fixtures import ml100k
    return ml100k().toPandas()


@pytest.fixture
def ml1m_data():
    from airship.tests.spark_fixtures import ml1m
    return ml1m().toPandas()


@pytest.mark.parametrize("dataset,splitter,model,ndcg", [
    (ml100k_data, random_splitter, sar_algo()[0], 0.8556358529344927),
    (ml1m_data, random_splitter, sar_algo()[0], 0.9053225659182779),
    (ml100k_data, random_splitter, sar_algo()[1], 0.8556358529344927),
    (ml1m_data, random_splitter, sar_algo()[1], 0.9053225659182779)
])
@pytest.mark.integration
def test_integration_sar(dataset, splitter, model, ndcg):
    sar_ref_algo(splitter(), model[0], dataset(), ndcg=ndcg)
