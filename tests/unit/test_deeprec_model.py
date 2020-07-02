# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.recommender.deeprec.deeprec_utils import (
    prepare_hparams,
    download_deeprec_resources,
)
from reco_utils.recommender.deeprec.models.xDeepFM import XDeepFMModel
from reco_utils.recommender.deeprec.models.dkn import DKN
from reco_utils.recommender.deeprec.io.iterator import FFMTextIterator
from reco_utils.recommender.deeprec.io.dkn_iterator import DKNTextIterator
from reco_utils.dataset.amazon_reviews import download_and_extract, data_preprocessing
from reco_utils.recommender.deeprec.models.sequential.sli_rec import SLI_RECModel
from reco_utils.recommender.deeprec.models.sequential.nextitnet import NextItNetModel
from reco_utils.recommender.deeprec.io.sequential_iterator import SequentialIterator
from reco_utils.recommender.deeprec.io.nextitnet_iterator import NextItNetIterator
from reco_utils.recommender.deeprec.models.graphrec.lightgcn import LightGCN
from reco_utils.recommender.deeprec.DataModel.ImplicitCF import ImplicitCF
from reco_utils.dataset import movielens
from reco_utils.dataset.python_splitters import python_stratified_split


@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.gpu
def test_xdeepfm_component_definition(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    hparams = prepare_hparams(yaml_file)
    model = XDeepFMModel(hparams, FFMTextIterator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None


@pytest.mark.gpu
def test_dkn_component_definition(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "dkn")
    yaml_file = os.path.join(data_path, "dkn.yaml")
    news_feature_file = os.path.join(data_path, r'doc_feature.txt')
    user_history_file = os.path.join(data_path, r'user_history.txt')
    wordEmb_file = os.path.join(data_path, r'word_embeddings_100.npy')
    entityEmb_file = os.path.join(data_path, r'TransE_entity2vec_100.npy')
    contextEmb_file = os.path.join(data_path, r'TransE_context2vec_100.npy')

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "dknresources.zip",
        )

    hparams = prepare_hparams(yaml_file,
                              news_feature_file=news_feature_file,
                              user_history_file=user_history_file,
                              wordEmb_file=wordEmb_file,
                              entityEmb_file=entityEmb_file,
                              contextEmb_file=contextEmb_file,
                              epochs=1,
                              learning_rate = 0.0001
                              )
    assert hparams is not None
    model = DKN(hparams, DKNTextIterator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None


@pytest.mark.gpu
def test_slirec_component_definition(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "slirec")
    yaml_file = os.path.join(
        resource_path,
        "..",
        "..",
        "reco_utils",
        "recommender",
        "deeprec",
        "config",
        "sli_rec.yaml",
    )
    yaml_file_nextitnet = os.path.join(
        resource_path,
        "..",
        "..",
        "reco_utils",
        "recommender",
        "deeprec",
        "config",
        "nextitnet.yaml",
    )
    train_file = os.path.join(data_path, r"train_data")

    if not os.path.exists(train_file):
        train_file = os.path.join(data_path, r"train_data")
        valid_file = os.path.join(data_path, r"valid_data")
        test_file = os.path.join(data_path, r"test_data")
        user_vocab = os.path.join(data_path, r"user_vocab.pkl")
        item_vocab = os.path.join(data_path, r"item_vocab.pkl")
        cate_vocab = os.path.join(data_path, r"category_vocab.pkl")

        reviews_name = "reviews_Movies_and_TV_5.json"
        meta_name = "meta_Movies_and_TV.json"
        reviews_file = os.path.join(data_path, reviews_name)
        meta_file = os.path.join(data_path, meta_name)
        valid_num_ngs = (
            4  # number of negative instances with a positive instance for validation
        )
        test_num_ngs = (
            9  # number of negative instances with a positive instance for testing
        )
        sample_rate = (
            0.01  # sample a small item set for training and testing here for example
        )

        input_files = [
            reviews_file,
            meta_file,
            train_file,
            valid_file,
            test_file,
            user_vocab,
            item_vocab,
            cate_vocab,
        ]
        download_and_extract(reviews_name, reviews_file)
        download_and_extract(meta_name, meta_file)
        data_preprocessing(
            *input_files,
            sample_rate=sample_rate,
            valid_num_ngs=valid_num_ngs,
            test_num_ngs=test_num_ngs
        )

    hparams = prepare_hparams(
        yaml_file, train_num_ngs=4
    )  # confirm the train_num_ngs when initializing a SLi_Rec model.
    model = SLI_RECModel(hparams, SequentialIterator)
    # nextitnet model
    hparams_nextitnet = prepare_hparams(yaml_file_nextitnet, train_num_ngs=4)
    model_nextitnet = NextItNetModel(hparams_nextitnet, NextItNetIterator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None

    assert model_nextitnet.logit is not None
    assert model_nextitnet.update is not None
    assert model_nextitnet.iterator is not None

@pytest.mark.gpu
def test_lightgcn_component_definition(resource_path):
    yaml_file = os.path.join(
        resource_path,
        "..",
        "..",
        "reco_utils",
        "recommender",
        "deeprec",
        "config",
        "lightgcn.yaml",
    )

    df = movielens.load_pandas_df(size="100k")
    train, test = python_stratified_split(df, ratio=0.75)

    data = ImplicitCF(train=train, test=test)

    embed_size = 64
    hparams = prepare_hparams(yaml_file, embed_size=embed_size)
    model = LightGCN(hparams, data)

    assert model.norm_adj is not None
    assert model.ua_embeddings.shape == [data.n_users, embed_size]
    assert model.ia_embeddings.shape == [data.n_items, embed_size]
    assert model.u_g_embeddings is not None
    assert model.pos_i_g_embeddings is not None
    assert model.neg_i_g_embeddings is not None
    assert model.batch_ratings is not None
    assert model.loss is not None
    assert model.opt is not None
