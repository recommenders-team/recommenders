import pytest

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
