# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import papermill as pm
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources  
from reco_utils.recommender.newsrec.models.base_model import BaseModel
from reco_utils.recommender.newsrec.models.nrms import NRMSModel
from reco_utils.recommender.newsrec.models.naml import NAMLModel
from reco_utils.recommender.newsrec.models.lstur import LSTURModel
from reco_utils.recommender.newsrec.models.npa import NPAModel
from reco_utils.recommender.newsrec.IO.news_iterator import NewsIterator
from reco_utils.recommender.newsrec.IO.naml_iterator import NAMLIterator


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.nrms
def test_model_nrms(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'nrms.yaml')
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'test.txt')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "nrms.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    assert hparams is not None

    iterator = NewsIterator  
    model = NRMSModel(hparams, iterator)

    assert model.run_eval(valid_file) is not None
    assert isinstance(model.fit(train_file, valid_file), BaseModel)

@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.naml
def test_model_naml(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'naml.yaml')
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'test.txt')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "naml.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    assert hparams is not None

    iterator = NAMLIterator
    model = NAMLModel(hparams, iterator)

    assert model.run_eval(valid_file) is not None
    assert isinstance(model.fit(train_file, valid_file), BaseModel)

@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.lstur
def test_model_lstur(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'lstur.yaml')
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'test.txt')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "lstur.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    assert hparams is not None

    iterator = NewsIterator
    model = LSTURModel(hparams, iterator)

    assert model.run_eval(valid_file) is not None
    assert isinstance(model.fit(train_file, valid_file), BaseModel)


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.newsrec
@pytest.mark.npa
def test_model_npa(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'npa.yaml')
    train_file = os.path.join(data_path, 'train.txt')
    valid_file = os.path.join(data_path, 'test.txt')
    wordEmb_file = os.path.join(data_path, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            data_path,
            "npa.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    assert hparams is not None

    iterator = NewsIterator
    model = NPAModel(hparams, iterator)

    assert model.run_eval(valid_file) is not None
    assert isinstance(model.fit(train_file, valid_file), BaseModel)