# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources  

from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.models.lstur import LSTURModel
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.IO.news_iterator import NewsIterator
from reco_utils.recommender.newsrec.IO.naml_iterator import NAMLIterator

@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))

@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.nrms
def test_nrms_component_definition(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'nrms.yaml')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "nrms.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    iterator = NewsIterator  
    model = NRMSModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None


@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.naml
def test_naml_component_definition(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'naml.yaml')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "naml.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    iterator = NAMLIterator   
    model = NAMLModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None


@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.npa
def test_npa_component_definition(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'npa.yaml')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "npa.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    iterator = NewsIterator
    model = NPAModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None

@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.lstur
def test_lstur_component_definition(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'lstur.yaml')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "lstur.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    iterator = NewsIterator  
    model = LSTURModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None