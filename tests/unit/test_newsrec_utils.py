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
@pytest.mark.newsrec
def test_prepare_hparams(must_exist_attributes, tmp):
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
    assert hasattr(hparams, must_exist_attributes)

@pytest.mark.gpu
@pytest.mark.newsrec
def test_load_yaml_file(tmp):
    data_path = tmp
    yaml_file = os.path.join(data_path, 'nrms.yaml')

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
def test_news_iterator(tmp):
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
    train_iterator = NewsIterator(hparams, hparams.npratio)
    test_iterator = NewsIterator(hparams, 0)

    assert train_iterator is not None
    for res in train_iterator.load_data_from_file(train_file):
        assert isinstance(res, dict)
        assert len(res) == 5
        assert res["impression_index_batch"].shape == (hparams.batch_size, 1)
        assert res["user_index_batch"].shape == (hparams.batch_size, 1)
        assert res["clicked_news_batch"].shape == (hparams.batch_size, hparams.his_size, hparams.doc_size)
        assert res["candidate_news_batch"].shape == (hparams.batch_size, hparams.npratio+1, hparams.doc_size)

    assert test_iterator is not None
    for res in test_iterator.load_data_from_file(valid_file):
        assert isinstance(res, dict)
        assert len(res) == 5
        assert res["impression_index_batch"].shape == (hparams.batch_size, 1)
        assert res["user_index_batch"].shape == (hparams.batch_size, 1)
        assert res["clicked_news_batch"].shape == (hparams.batch_size, hparams.his_size, hparams.doc_size)
        assert res["candidate_news_batch"].shape == (hparams.batch_size, 1, hparams.doc_size)



@pytest.mark.gpu
@pytest.mark.newsrec
def test_naml_iterator(tmp):
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
    train_iterator = NAMLIterator(hparams, hparams.npratio)
    test_iterator = NAMLIterator(hparams, 0)

    assert train_iterator is not None
    for res in train_iterator.load_data_from_file(train_file):
        assert isinstance(res, dict)
        assert len(res) == 11
        
        assert res["impression_index_batch"].shape == (hparams.batch_size, 1)
        assert res["user_index_batch"].shape == (hparams.batch_size, 1)
        assert res["clicked_title_batch"].shape == (hparams.batch_size, hparams.his_size, hparams.title_size)
        assert res["clicked_body_batch"].shape == (hparams.batch_size, hparams.his_size, hparams.body_size)
        assert res["clicked_vert_batch"].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res["clicked_subvert_batch"].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res["candidate_title_batch"].shape == (hparams.batch_size, hparams.npratio+1, hparams.title_size)
        assert res["candidate_body_batch"].shape == (hparams.batch_size, hparams.npratio+1, hparams.body_size)
        assert res["candidate_vert_batch"].shape == (hparams.batch_size, hparams.npratio+1, 1)
        assert res["candidate_subvert_batch"].shape == (hparams.batch_size, hparams.npratio+1, 1)

    assert test_iterator is not None
    for res in test_iterator.load_data_from_file(valid_file):
        assert isinstance(res, dict)
        assert len(res) == 11
        assert res["impression_index_batch"].shape == (hparams.batch_size, 1)
        assert res["user_index_batch"].shape == (hparams.batch_size, 1)
        assert res["clicked_title_batch"].shape == (hparams.batch_size, hparams.his_size, hparams.title_size)
        assert res["clicked_body_batch"].shape == (hparams.batch_size, hparams.his_size, hparams.body_size)
        assert res["clicked_vert_batch"].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res["clicked_subvert_batch"].shape == (hparams.batch_size, hparams.his_size, 1)
        assert res["candidate_title_batch"].shape == (hparams.batch_size, 1, hparams.title_size)
        assert res["candidate_body_batch"].shape == (hparams.batch_size, 1, hparams.body_size)
        assert res["candidate_vert_batch"].shape == (hparams.batch_size, 1, 1)
        assert res["candidate_subvert_batch"].shape == (hparams.batch_size, 1, 1)

