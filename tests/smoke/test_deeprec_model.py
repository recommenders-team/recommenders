# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import papermill as pm
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources, prepare_hparams
from reco_utils.recommender.deeprec.models.base_model import BaseModel
from reco_utils.recommender.deeprec.models.xDeepFM import XDeepFMModel
from reco_utils.recommender.deeprec.models.dkn import DKN
from reco_utils.recommender.deeprec.IO.iterator import FFMTextIterator
from reco_utils.recommender.deeprec.IO.dkn_iterator import DKNTextIterator


@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_xdeepfm(resource_path):
    data_path = os.path.join(resource_path, '../resources/deeprec/xdeepfm')
    yaml_file = os.path.join(data_path, r'xDeepFM.yaml')
    data_file = os.path.join(data_path, r'sample_FFM_data.txt')
    output_file = os.path.join(data_path, r'output.txt')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path, 'xdeepfmresources.zip')

    hparams = prepare_hparams(yaml_file, learning_rate=0.01)
    assert hparams is not None

    input_creator = FFMTextIterator
    model = XDeepFMModel(hparams, input_creator)

    assert model.run_eval(data_file) is not None
    assert isinstance(model.fit(data_file,data_file), BaseModel)
    assert model.predict(data_file, output_file) is not None
    del model

    
@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_dkn(resource_path):
    data_path = os.path.join(resource_path, '../resources/deeprec/dkn')
    yaml_file = os.path.join(data_path, r'dkn.yaml')
    train_file = os.path.join(data_path, r'final_test_with_entity.txt')
    valid_file = os.path.join(data_path, r'final_test_with_entity.txt')
    wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
    entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(r'https://recodatasets.blob.core.windows.net/deeprec/', data_path, 'dknresources.zip')

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file,
                              entityEmb_file=entityEmb_file, epochs=1, learning_rate=0.0001)
    input_creator = DKNTextIterator
    model = DKN(hparams, input_creator)

    assert(isinstance(model.fit(train_file, valid_file), BaseModel))
    assert model.run_eval(valid_file) is not None
    del model

  
