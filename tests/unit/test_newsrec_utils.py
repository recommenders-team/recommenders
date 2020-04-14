# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import tensorflow as tf
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams, load_yaml
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources  

from reco_utils.recommender.newsrec.IO.news_iterator import NewsTrainIterator, NewsTestIterator
from reco_utils.recommender.newsrec.IO.naml_iterator import NAMLTrainIterator, NAMLTestIterator

@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "must_exist_attributes", ["word_size", "data_format", "word_emb_dim"]
)
@pytest.mark.gpu
@pytest.mark.newsrec
def test_prepare_hparams(must_exist_attributes, resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "newsrec", "nrms")
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
    assert hasattr(hparams, must_exist_attributes)

@pytest.mark.gpu
@pytest.mark.newsrec
def test_load_yaml_file(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "newsrec", "nrms")
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
    config = load_yaml(yaml_file)
    assert config is not None

@pytest.mark.gpu
@pytest.mark.newsrec
def test_NewsTrain_iterator(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "newsrec", "nrms")
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
    iterator = NewsTrainIterator(hparams)

    assert iterator is not None
    for res in iterator.load_data_from_file(train_file):
        assert isinstance(res, tuple)
        assert len(res[0]) == 4
        assert res[0][0].shape == (hparams.batch_size, 1)
        assert res[0][1].shape == (hparams.batch_size, 1)
        assert res[0][2].shape == (hparams.batch_size, hparams.his_size, hparams.doc_size)
        assert res[0][3].shape == (hparams.batch_size, hparams.npratio+1, hparams.doc_size)

@pytest.mark.gpu
@pytest.mark.newsrec
def test_NewsTest_iterator(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "newsrec", "nrms")
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
    iterator = NewsTestIterator(hparams)

    assert iterator is not None
    for res in iterator.load_data_from_file(valid_file):
        assert isinstance(res, tuple)
        assert len(res[0]) == 4
        assert res[0][0].shape == (hparams.batch_size, 1)
        assert res[0][1].shape == (hparams.batch_size, 1)
        assert res[0][2].shape == (hparams.batch_size, hparams.his_size, hparams.doc_size)
        assert res[0][3].shape == (hparams.batch_size, hparams.doc_size)


@pytest.mark.gpu
@pytest.mark.newsrec
def test_NAMLTrain_iterator(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "newsrec", "naml")
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
    iterator = NAMLTrainIterator(hparams)

    assert iterator is not None
    for res in iterator.load_data_from_file(train_file):
        assert isinstance(res, tuple)
        assert len(res[0]) == 10
        assert res[0][0].shape == (hparams.batch_size, 1)
        assert res[0][1].shape == (hparams.batch_size, 1)
        assert res[0][2].shape == (hparams.batch_size, hparams.his_size, hparams.title_size)
        assert res[0][3].shape == (hparams.batch_size, hparams.his_size, hparams.body_size)
        assert res[0][4].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res[0][5].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res[0][6].shape == (hparams.batch_size, hparams.npratio+1, hparams.title_size)
        assert res[0][7].shape == (hparams.batch_size, hparams.npratio+1, hparams.body_size)
        assert res[0][8].shape == (hparams.batch_size, hparams.npratio+1, 1)
        assert res[0][9].shape == (hparams.batch_size, hparams.npratio+1, 1)

@pytest.mark.gpu
@pytest.mark.newsrec
def test_NAMLTrain_iterator(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "newsrec", "naml")
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
    iterator = NAMLTestIterator(hparams)

    assert iterator is not None
    for res in iterator.load_data_from_file(valid_file):
        assert isinstance(res, tuple)
        assert len(res[0]) == 10
        assert res[0][0].shape == (hparams.batch_size, 1)
        assert res[0][1].shape == (hparams.batch_size, 1)
        assert res[0][2].shape == (hparams.batch_size, hparams.his_size, hparams.title_size)
        assert res[0][3].shape == (hparams.batch_size, hparams.his_size, hparams.body_size)
        assert res[0][4].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res[0][5].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res[0][6].shape == (hparams.batch_size, hparams.title_size)
        assert res[0][7].shape == (hparams.batch_size, hparams.body_size)
        assert res[0][8].shape == (hparams.batch_size, 1)
        assert res[0][9].shape == (hparams.batch_size, 1)