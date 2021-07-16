# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.datasets import movielens
from reco_utils.datasets.amazon_reviews import download_and_extract, data_preprocessing
from reco_utils.datasets.python_splitters import python_stratified_split

try:
    from reco_utils.models.deeprec.DataModel.ImplicitCF import ImplicitCF
    from reco_utils.models.deeprec.deeprec_utils import (
        prepare_hparams,
        download_deeprec_resources,
    )
    from reco_utils.models.deeprec.io.iterator import FFMTextIterator
    from reco_utils.models.deeprec.io.dkn_item2item_iterator import (
        DKNItem2itemTextIterator,
    )
    from reco_utils.models.deeprec.io.dkn_iterator import DKNTextIterator
    from reco_utils.models.deeprec.io.nextitnet_iterator import NextItNetIterator
    from reco_utils.models.deeprec.io.sequential_iterator import SequentialIterator
    from reco_utils.models.deeprec.models.dkn import DKN
    from reco_utils.models.deeprec.models.dkn_item2item import DKNItem2Item
    from reco_utils.models.deeprec.models.graphrec.lightgcn import LightGCN
    from reco_utils.models.deeprec.models.sequential.nextitnet import (
        NextItNetModel,
    )
    from reco_utils.models.deeprec.models.sequential.sli_rec import SLI_RECModel
    from reco_utils.models.deeprec.models.sequential.sum import SUMModel
    from reco_utils.models.deeprec.models.xDeepFM import XDeepFMModel
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.gpu
def test_xdeepfm_component_definition(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    hparams = prepare_hparams(yaml_file)
    model = XDeepFMModel(hparams, FFMTextIterator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None


@pytest.mark.gpu
@pytest.fixture(scope="module")
def dkn_files(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    yaml_file = os.path.join(data_path, "dkn.yaml")
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
    return (
        data_path,
        yaml_file,
        news_feature_file,
        user_history_file,
        wordEmb_file,
        entityEmb_file,
        contextEmb_file,
    )


@pytest.mark.gpu
def test_dkn_component_definition(dkn_files):
    # Load params from fixture
    (
        _,
        yaml_file,
        news_feature_file,
        user_history_file,
        wordEmb_file,
        entityEmb_file,
        contextEmb_file,
    ) = dkn_files

    # Test DKN model
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
    assert hparams is not None

    model = DKN(hparams, DKNTextIterator)
    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None


@pytest.mark.gpu
def test_dkn_item2item_component_definition(dkn_files):
    # Load params from fixture
    (
        data_path,
        yaml_file,
        news_feature_file,
        _,
        wordEmb_file,
        entityEmb_file,
        contextEmb_file,
    ) = dkn_files

    # Test DKN's item2item version
    hparams = prepare_hparams(
        yaml_file,
        news_feature_file=news_feature_file,
        wordEmb_file=wordEmb_file,
        entityEmb_file=entityEmb_file,
        contextEmb_file=contextEmb_file,
        epochs=1,
        is_clip_norm=True,
        max_grad_norm=0.5,
        his_size=20,
        MODEL_DIR=os.path.join(data_path, "save_models"),
        use_entity=True,
        use_context=True,
    )
    assert hparams is not None

    hparams.neg_num = 9
    model_item2item = DKNItem2Item(hparams, DKNItem2itemTextIterator)
    assert model_item2item.pred_logits is not None
    assert model_item2item.update is not None
    assert model_item2item.iterator is not None


@pytest.mark.gpu
@pytest.fixture(scope="module")
def sequential_files(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "slirec")
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

    return (
        data_path,
        user_vocab,
        item_vocab,
        cate_vocab,
    )


@pytest.mark.gpu
def test_slirec_component_definition(sequential_files, deeprec_config_path):
    yaml_file = os.path.join(deeprec_config_path, "sli_rec.yaml")
    data_path, user_vocab, item_vocab, cate_vocab = sequential_files

    hparams = prepare_hparams(
        yaml_file,
        train_num_ngs=4,
        embed_l2=0.0,
        layer_l2=0.0,
        learning_rate=0.001,
        epochs=1,
        MODEL_DIR=os.path.join(data_path, "model"),
        SUMMARIES_DIR=os.path.join(data_path, "summary"),
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        cate_vocab=cate_vocab,
        need_sample=True,
    )
    assert hparams is not None

    model = SLI_RECModel(hparams, SequentialIterator)
    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None


@pytest.mark.gpu
def test_nextitnet_component_definition(sequential_files, deeprec_config_path):
    yaml_file_nextitnet = os.path.join(deeprec_config_path, "nextitnet.yaml")
    data_path, user_vocab, item_vocab, cate_vocab = sequential_files

    # NextItNet model
    hparams_nextitnet = prepare_hparams(
        yaml_file_nextitnet,
        train_num_ngs=4,
        embed_l2=0.0,
        layer_l2=0.0,
        learning_rate=0.001,
        epochs=1,
        MODEL_DIR=os.path.join(data_path, "model"),
        SUMMARIES_DIR=os.path.join(data_path, "summary"),
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        cate_vocab=cate_vocab,
        need_sample=True,
    )
    assert hparams_nextitnet is not None

    model_nextitnet = NextItNetModel(hparams_nextitnet, NextItNetIterator)
    assert model_nextitnet.logit is not None
    assert model_nextitnet.update is not None
    assert model_nextitnet.iterator is not None


@pytest.mark.gpu
def test_sum_component_definition(sequential_files, deeprec_config_path):
    yaml_file_sum = os.path.join(deeprec_config_path, "sum.yaml")
    data_path, user_vocab, item_vocab, cate_vocab = sequential_files

    # SUM model
    hparams_sum = prepare_hparams(
        yaml_file_sum,
        train_num_ngs=4,
        embed_l2=0.0,
        layer_l2=0.0,
        learning_rate=0.001,
        epochs=1,
        MODEL_DIR=os.path.join(data_path, "model"),
        SUMMARIES_DIR=os.path.join(data_path, "summary"),
        user_vocab=user_vocab,
        item_vocab=item_vocab,
        cate_vocab=cate_vocab,
        need_sample=True,
    )
    assert hparams_sum is not None

    model_sum = SUMModel(hparams_sum, SequentialIterator)
    assert model_sum.logit is not None
    assert model_sum.update is not None
    assert model_sum.iterator is not None


@pytest.mark.gpu
def test_lightgcn_component_definition(deeprec_config_path):
    yaml_file = os.path.join(deeprec_config_path, "lightgcn.yaml")

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
