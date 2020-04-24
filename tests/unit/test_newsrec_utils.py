# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import tensorflow as tf
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams, load_yaml
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources  

from reco_utils.recommender.newsrec.IO.news_iterator import NewsIterator
from reco_utils.recommender.newsrec.IO.naml_iterator import NAMLIterator

@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "must_exist_attributes", ["word_size", "data_format", "word_emb_dim"]
)
@pytest.mark.gpu
def test_prepare_hparams(must_exist_attributes, tmp):
    yaml_file = os.path.join(tmp, 'nrms.yaml')
    train_file = os.path.join(tmp, 'train.txt')
    valid_file = os.path.join(tmp, 'test.txt')
    wordEmb_file = os.path.join(tmp, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            tmp,
            "nrms.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1)
    assert hasattr(hparams, must_exist_attributes)

@pytest.mark.gpu
def test_load_yaml_file(tmp):
    yaml_file = os.path.join(tmp, 'nrms.yaml')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            tmp,
            "nrms.zip",
        )
    config = load_yaml(yaml_file)
    assert config is not None

@pytest.mark.gpu
def test_news_iterator(tmp):
    yaml_file = os.path.join(tmp, 'nrms.yaml')
    train_file = os.path.join(tmp, 'train.txt')
    valid_file = os.path.join(tmp, 'test.txt')
    wordEmb_file = os.path.join(tmp, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            tmp,
            "nrms.zip",
        )
    
    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1, batch_size=512)
    train_iterator = NewsIterator(hparams, hparams.npratio)
    test_iterator = NewsIterator(hparams, 0)

    assert train_iterator is not None
    for res in train_iterator.load_data_from_file(train_file):
        assert isinstance(res, dict)
        assert len(res) == 5
        break
    
    assert test_iterator is not None
    for res in test_iterator.load_data_from_file(valid_file):
        assert isinstance(res, dict)
        assert len(res) == 5
        break


@pytest.mark.gpu
def test_naml_iterator(tmp):
    yaml_file = os.path.join(tmp, 'naml.yaml')
    train_file = os.path.join(tmp, 'train.txt')
    valid_file = os.path.join(tmp, 'test.txt')
    wordEmb_file = os.path.join(tmp, 'embedding.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/newsrec/",
            tmp,
            "naml.zip",
        )
    
    hparams = prepare_hparams(yaml_file, wordEmb_file=wordEmb_file, epochs=1, batch_size=1024)
    train_iterator = NAMLIterator(hparams, hparams.npratio)
    test_iterator = NAMLIterator(hparams, 0)

    assert train_iterator is not None
    for res in train_iterator.load_data_from_file(train_file):
        assert isinstance(res, dict)
        assert len(res) == 11
        break
        
    assert test_iterator is not None
    for res in test_iterator.load_data_from_file(valid_file):
        assert isinstance(res, dict)
        assert len(res) == 11
        break
