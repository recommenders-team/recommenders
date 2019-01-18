
import pytest
import os
from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.models.xDeepFM import *
from reco_utils.recommender.deeprec.IO.iterator import *

@pytest.mark.deeprec
def test_model_component_definition():
    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(cur_file_path, r'../resources/deeprec/xDeepFM.yaml')
    hparams = prepare_hparams(yaml_file)
    input_creator = FFMTextIterator
    model = XDeepFMModel(hparams, input_creator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None

# if __name__ == '__main__':
#     cur_file_path = os.path.dirname(os.path.realpath(__file__))
#     yaml_file = os.path.join(cur_file_path, r'../resources/deeprec/xDeepFM.yaml')
#     print(os.path.dirname(yaml_file))
#     print(os.path.basename(yaml_file))
#     print(os.path.abspath(yaml_file))