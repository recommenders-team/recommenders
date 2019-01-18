
import pytest
import os
from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.IO.iterator import *
import tensorflow as tf

@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))

@pytest.mark.deeprec
@pytest.mark.parametrize("must_exist_attributes", [
    "FEATURE_COUNT", "data_format", "dim"
])
def test_prepare_hparams(must_exist_attributes):
    hparams = prepare_hparams('')
    assert hasattr(hparams, must_exist_attributes)

@pytest.mark.deeprec
def test_load_yaml_file(resource_path):
    yaml_file = os.path.join(resource_path, r'../resources/deeprec/xDeepFM.yaml')
    config = load_yaml_file(yaml_file)
    assert config is not None

@pytest.mark.deeprec
def test_FFM_iterator(resource_path):
    data_file = os.path.join(resource_path, r'../resources/deeprec/sample_FFM_data.txt')
    iterator = FFMTextIterator(10000, 10, tf.Graph())
    assert iterator is not None
    for res in iterator.load_data_from_file(data_file):
        assert isinstance(res, dict)


# if __name__ == '__main__':
#     cur_file_path = os.path.dirname(os.path.realpath(__file__))
#     yaml_file = os.path.join(cur_file_path, r'../resources/deeprec/xDeepFM.yaml')
#     print(os.path.dirname(yaml_file))
#     print(os.path.basename(yaml_file))
#     print(os.path.abspath(yaml_file))