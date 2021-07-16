# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import papermill as pm
import pytest

try:
    import tensorflow as tf
    from reco_utils.models.deeprec.deeprec_utils import (
        download_deeprec_resources,
        prepare_hparams,
    )
    from reco_utils.models.deeprec.models.base_model import BaseModel
    from reco_utils.models.deeprec.models.xDeepFM import XDeepFMModel
    from reco_utils.models.deeprec.models.dkn import DKN
    from reco_utils.models.deeprec.io.iterator import FFMTextIterator
    from reco_utils.models.deeprec.io.dkn_iterator import DKNTextIterator
    from reco_utils.models.deeprec.io.sequential_iterator import SequentialIterator
    from reco_utils.models.deeprec.models.sequential.sli_rec import SLI_RECModel
    from reco_utils.models.deeprec.models.sequential.sum import SUMModel
    from reco_utils.datasets.amazon_reviews import (
        download_and_extract,
        data_preprocessing,
    )
    from reco_utils.models.deeprec.models.graphrec.lightgcn import LightGCN
    from reco_utils.models.deeprec.DataModel.ImplicitCF import ImplicitCF
    from reco_utils.datasets import movielens
    from reco_utils.datasets.python_splitters import python_stratified_split
except ImportError:
    pass  # disable error while collecting tests for non-gpu environments


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_FFM_iterator(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    data_file = os.path.join(data_path, "sample_FFM_data.txt")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    hparams = prepare_hparams(yaml_file)
    iterator = FFMTextIterator(hparams, tf.Graph())
    assert iterator is not None
    for res in iterator.load_data_from_file(data_file):
        assert isinstance(res, tuple)


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_xdeepfm(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    data_file = os.path.join(data_path, "sample_FFM_data.txt")
    output_file = os.path.join(data_path, "output.txt")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    hparams = prepare_hparams(yaml_file, learning_rate=0.01)
    assert hparams is not None

    input_creator = FFMTextIterator
    model = XDeepFMModel(hparams, input_creator)

    assert model.run_eval(data_file) is not None
    assert isinstance(model.fit(data_file, data_file), BaseModel)
    assert model.predict(data_file, output_file) is not None


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_dkn(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    yaml_file = os.path.join(data_path, r"dkn.yaml")
    train_file = os.path.join(data_path, r"train_mind_demo.txt")
    valid_file = os.path.join(data_path, r"valid_mind_demo.txt")
    test_file = os.path.join(data_path, r"test_mind_demo.txt")
    news_feature_file = os.path.join(data_path, r"doc_feature.txt")
    user_history_file = os.path.join(data_path, r"user_history.txt")
    wordEmb_file = os.path.join(data_path, r"word_embeddings_100.npy")
    entityEmb_file = os.path.join(data_path, r"TransE_entity2vec_100.npy")
    contextEmb_file = os.path.join(data_path, r"TransE_context2vec_100.npy")

    download_deeprec_resources(
        "https://recodatasets.z20.web.core.windows.net/deeprec/",
        data_path,
        "mind-demo.zip",
    )

    hparams = prepare_hparams(
        yaml_file,
        news_feature_file=news_feature_file,
        user_history_file=user_history_file,
        wordEmb_file=wordEmb_file,
        entityEmb_file=entityEmb_file,
        contextEmb_file=contextEmb_file,
        epochs=1,
        learning_rate=0.0001,
    )
    input_creator = DKNTextIterator
    model = DKN(hparams, input_creator)

    assert isinstance(model.fit(train_file, valid_file), BaseModel)
    assert model.run_eval(valid_file) is not None


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
@pytest.mark.sequential
def test_model_slirec(deeprec_resource_path, deeprec_config_path):
    data_path = os.path.join(deeprec_resource_path, "slirec")
    yaml_file = os.path.join(deeprec_config_path, "sli_rec.yaml")
    train_file = os.path.join(data_path, r"train_data")
    valid_file = os.path.join(data_path, r"valid_data")
    test_file = os.path.join(data_path, r"test_data")
    output_file = os.path.join(data_path, "output.txt")
    train_num_ngs = (
        4  # number of negative instances with a positive instance for training
    )
    valid_num_ngs = (
        4  # number of negative instances with a positive instance for validation
    )
    test_num_ngs = (
        9  # number of negative instances with a positive instance for testing
    )

    if not os.path.exists(train_file):
        user_vocab = os.path.join(data_path, r"user_vocab.pkl")
        item_vocab = os.path.join(data_path, r"item_vocab.pkl")
        cate_vocab = os.path.join(data_path, r"category_vocab.pkl")
        reviews_name = "reviews_Movies_and_TV_5.json"
        meta_name = "meta_Movies_and_TV.json"
        reviews_file = os.path.join(data_path, reviews_name)
        meta_file = os.path.join(data_path, meta_name)
        sample_rate = (
            0.005  # sample a small item set for training and testing here for example
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
        yaml_file, learning_rate=0.01, epochs=3, train_num_ngs=train_num_ngs
    )  # confirm train_num_ngs before initializing a SLi_Rec model.
    assert hparams is not None

    input_creator = SequentialIterator
    model = SLI_RECModel(hparams, input_creator)
    assert model.run_eval(valid_file, num_ngs=valid_num_ngs) is not None
    assert isinstance(
        model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs), BaseModel
    )
    assert model.predict(test_file, output_file) is not None


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
@pytest.mark.sequential
def test_model_sum(deeprec_resource_path, deeprec_config_path):
    data_path = os.path.join(deeprec_resource_path, "slirec")
    yaml_file = os.path.join(deeprec_config_path, "sum.yaml")
    train_file = os.path.join(data_path, r"train_data")
    valid_file = os.path.join(data_path, r"valid_data")
    test_file = os.path.join(data_path, r"test_data")
    output_file = os.path.join(data_path, "output.txt")
    train_num_ngs = (
        4  # number of negative instances with a positive instance for training
    )
    valid_num_ngs = (
        4  # number of negative instances with a positive instance for validation
    )
    test_num_ngs = (
        9  # number of negative instances with a positive instance for testing
    )

    if not os.path.exists(train_file):
        user_vocab = os.path.join(data_path, r"user_vocab.pkl")
        item_vocab = os.path.join(data_path, r"item_vocab.pkl")
        cate_vocab = os.path.join(data_path, r"category_vocab.pkl")
        reviews_name = "reviews_Movies_and_TV_5.json"
        meta_name = "meta_Movies_and_TV.json"
        reviews_file = os.path.join(data_path, reviews_name)
        meta_file = os.path.join(data_path, meta_name)
        sample_rate = (
            0.005  # sample a small item set for training and testing here for example
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
        yaml_file, learning_rate=0.01, epochs=1, train_num_ngs=train_num_ngs
    )
    assert hparams is not None

    input_creator = SequentialIterator
    model = SUMModel(hparams, input_creator)
    assert model.run_eval(valid_file, num_ngs=valid_num_ngs) is not None
    assert isinstance(
        model.fit(train_file, valid_file, valid_num_ngs=valid_num_ngs), BaseModel
    )
    assert model.predict(valid_file, output_file) is not None


@pytest.mark.smoke
@pytest.mark.gpu
@pytest.mark.deeprec
def test_model_lightgcn(deeprec_resource_path, deeprec_config_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    yaml_file = os.path.join(deeprec_config_path, "lightgcn.yaml")
    user_file = os.path.join(data_path, r"user_embeddings.csv")
    item_file = os.path.join(data_path, r"item_embeddings.csv")

    df = movielens.load_pandas_df(size="100k")
    train, test = python_stratified_split(df, ratio=0.75)

    data = ImplicitCF(train=train, test=test)

    hparams = prepare_hparams(yaml_file, epochs=1)
    model = LightGCN(hparams, data)

    assert model.run_eval() is not None
    model.fit()
    assert model.recommend_k_items(test) is not None
    model.infer_embedding(user_file, item_file)
    assert os.path.getsize(user_file) != 0
    assert os.path.getsize(item_file) != 0
