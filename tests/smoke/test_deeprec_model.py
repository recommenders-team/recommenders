
import pytest
import os
from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.models.base_model import *
from reco_utils.recommender.deeprec.models.xDeepFM import *
from reco_utils.recommender.deeprec.models.dkn import *
from reco_utils.recommender.deeprec.IO.iterator import *
from reco_utils.recommender.deeprec.IO.dkn_iterator import *

@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))

@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_xdeepfm(resource_path):
    yaml_file = os.path.join(resource_path, r'../resources/deeprec/xDeepFM.yaml')
    data_file = os.path.join(resource_path, r'../resources/deeprec/sample_FFM_data.txt')
    output_file = os.path.join(resource_path, r'../resources/deeprec/output.txt')

    hparams = prepare_hparams(yaml_file, learning_rate=0.01)
    assert hparams is not None

    input_creator = FFMTextIterator
    model = XDeepFMModel(hparams, input_creator)

    assert model.run_eval(data_file) is not None
    assert isinstance(model.fit(data_file,data_file), BaseModel)
    assert model.predict(data_file, output_file) is not None

@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_dkn(resource_path):
    yaml_file = os.path.join(resource_path, r'../resources/deeprec/dkn/dkn.yaml')
    train_file = os.path.join(resource_path, r'../resources/deeprec/dkn/final_test_with_entity.txt')
    valid_file = os.path.join(resource_path, r'../resources/deeprec/dkn/final_test_with_entity.txt')
    wordEmb_file = os.path.join(resource_path, r'../resources/deeprec/dkn/word_embeddings_100.npy')
    entityEmb_file = os.path.join(resource_path, r'../resources/deeprec/dkn/TransE_entity2vec_100.npy')

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file,
                              entityEmb_file=entityEmb_file, epochs=2, learning_rate=0.0001)
    input_creator = DKNTextIterator
    model = DKN(hparams, input_creator)

    assert(isinstance(model.fit(train_file, valid_file), BaseModel))
    assert model.run_eval(valid_file) is not None


# if __name__ == '__main__':
#     cur_file_path = os.path.dirname(os.path.realpath(__file__))
#     yaml_file = os.path.join(cur_file_path, r'../resources/deeprec/xDeepFM.yaml')
#     print(os.path.dirname(yaml_file))
#     print(os.path.basename(yaml_file))
#     print(os.path.abspath(yaml_file))