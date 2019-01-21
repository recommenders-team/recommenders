
import pytest
import os
from reco_utils.recommender.deeprec.deeprec_utils import *
from reco_utils.recommender.deeprec.models.xDeepFM import *
from reco_utils.recommender.deeprec.models.dkn import *
from reco_utils.recommender.deeprec.IO.iterator import *
from reco_utils.recommender.deeprec.IO.dkn_iterator import *

@pytest.mark.gpu
@pytest.mark.deeprec
def test_xdeepfm_component_definition():
    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(cur_file_path, r'../resources/deeprec/xDeepFM.yaml')
    hparams = prepare_hparams(yaml_file)
    input_creator = FFMTextIterator
    model = XDeepFMModel(hparams, input_creator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None


@pytest.mark.gpu
@pytest.mark.deeprec
def test_dkn_component_definition():
    cur_file_path = os.path.dirname(os.path.realpath(__file__))
    yaml_file = os.path.join(cur_file_path, r'../resources/deeprec/dkn/dkn.yaml')
    wordEmb_file = os.path.join(cur_file_path, r'../resources/deeprec/dkn/word_embeddings_100.npy')
    entityEmb_file = os.path.join(cur_file_path, r'../resources/deeprec/dkn/TransE_entity2vec_100.npy')

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file,
                              entityEmb_file=entityEmb_file, epochs=5, learning_rate=0.0001)
    assert hparams is not None
    input_creator = DKNTextIterator
    model = DKN(hparams, input_creator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None
