# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pytest

from recommenders.datasets import movielens
from recommenders.datasets.amazon_reviews import (
    download_and_extract,
    data_preprocessing,
)
from recommenders.datasets.python_splitters import python_stratified_split

try:
    from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
    from recommenders.models.deeprec.deeprec_utils import (
        prepare_hparams,
        download_deeprec_resources,
    )
    from recommenders.models.deeprec.io.iterator import FFMTextIterator
    from recommenders.models.deeprec.io.dkn_item2item_iterator import (
        DKNItem2itemTextIterator,
    )
    from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator
    from recommenders.models.deeprec.io.nextitnet_iterator import NextItNetIterator
    from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
    from recommenders.models.deeprec.models.dkn import DKN
    from recommenders.models.deeprec.models.dkn_item2item import DKNItem2Item
    from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
    from recommenders.models.deeprec.models.sequential.nextitnet import (
        NextItNetModel,
    )
    from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel
    from recommenders.models.deeprec.models.sequential.sum import SUMModel
    from recommenders.models.deeprec.models.xDeepFM import XDeepFMModel
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.gpu
@pytest.fixture(scope="module")
def dkn_files(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    yaml_file = os.path.join(data_path, "dkn.yaml")
    news_feature_file = os.path.join(data_path, "doc_feature.txt")
    user_history_file = os.path.join(data_path, "user_history.txt")
    wordEmb_file = os.path.join(data_path, "word_embeddings_100.npy")
    entityEmb_file = os.path.join(data_path, "TransE_entity2vec_100.npy")
    contextEmb_file = os.path.join(data_path, "TransE_context2vec_100.npy")

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
@pytest.fixture(scope="module")
def sequential_files(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "slirec")
    train_file = os.path.join(data_path, "train_data")
    valid_file = os.path.join(data_path, "valid_data")
    test_file = os.path.join(data_path, "test_data")
    user_vocab = os.path.join(data_path, "user_vocab.pkl")
    item_vocab = os.path.join(data_path, "item_vocab.pkl")
    cate_vocab = os.path.join(data_path, "category_vocab.pkl")

    reviews_name = "reviews_Movies_and_TV_5.json"
    meta_name = "meta_Movies_and_TV.json"
    reviews_file = os.path.join(data_path, reviews_name)
    meta_file = os.path.join(data_path, meta_name)

    # number of negative instances with a positive instance for validation
    valid_num_ngs = 4
    # number of negative instances with a positive instance for testing
    test_num_ngs = 9
    # sample a small item set for training and testing here for example
    sample_rate = 0.01

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
    assert model.hparams is not None
    assert model.hparams.model_type == "xDeepFM"
    assert model.hparams.epochs == 50
    assert model.hparams.batch_size == 128
    assert model.hparams.learning_rate == 0.0005
    assert model.hparams.loss == "log_loss"
    assert model.hparams.optimizer == "adam"


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

    model = DKN(hparams, DKNTextIterator)
    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None
    assert model.hparams is not None
    assert model.hparams.model_type == "dkn"
    assert model.hparams.epochs == 1
    assert model.hparams.batch_size == 100
    assert model.hparams.learning_rate == 0.0001
    assert model.hparams.loss == "log_loss"
    assert model.hparams.optimizer == "adam"


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

    hparams.neg_num = 9
    model_item2item = DKNItem2Item(hparams, DKNItem2itemTextIterator)
    assert model_item2item.pred_logits is not None
    assert model_item2item.update is not None
    assert model_item2item.iterator is not None
    assert model_item2item.hparams is not None
    assert model_item2item.hparams.model_type == "dkn"
    assert model_item2item.hparams.epochs == 1
    assert model_item2item.hparams.batch_size == 100
    assert model_item2item.hparams.learning_rate == 0.0005
    assert model_item2item.hparams.loss == "log_loss"
    assert model_item2item.hparams.optimizer == "adam"
    assert model_item2item.hparams.max_grad_norm == 0.5
    assert model_item2item.hparams.his_size == 20


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

    model = SLI_RECModel(hparams, SequentialIterator)
    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None
    assert model.hparams is not None
    assert model.hparams.model_type == "sli_rec"
    assert model.hparams.epochs == 1
    assert model.hparams.batch_size == 400
    assert model.hparams.learning_rate == 0.001
    assert model.hparams.loss == "softmax"
    assert model.hparams.optimizer == "adam"
    assert model.hparams.train_num_ngs == 4
    assert model.hparams.embed_l2 == 0.0
    assert model.hparams.layer_l2 == 0.0
    assert model.hparams.need_sample is True


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

    model_nextitnet = NextItNetModel(hparams_nextitnet, NextItNetIterator)
    assert model_nextitnet.logit is not None
    assert model_nextitnet.update is not None
    assert model_nextitnet.iterator is not None
    assert model_nextitnet.hparams is not None
    assert model_nextitnet.hparams.model_type == "NextItNet"
    assert model_nextitnet.hparams.epochs == 1
    assert model_nextitnet.hparams.batch_size == 400
    assert model_nextitnet.hparams.learning_rate == 0.001
    assert model_nextitnet.hparams.loss == "softmax"
    assert model_nextitnet.hparams.optimizer == "adam"
    assert model_nextitnet.hparams.train_num_ngs == 4
    assert model_nextitnet.hparams.embed_l2 == 0.0
    assert model_nextitnet.hparams.layer_l2 == 0.0
    assert model_nextitnet.hparams.need_sample is True


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

    model_sum = SUMModel(hparams_sum, SequentialIterator)
    assert model_sum.logit is not None
    assert model_sum.update is not None
    assert model_sum.iterator is not None
    assert model_sum.hparams is not None
    assert model_sum.hparams.model_type == "SUM"
    assert model_sum.hparams.epochs == 1
    assert model_sum.hparams.batch_size == 400
    assert model_sum.hparams.learning_rate == 0.001
    assert model_sum.hparams.loss == "softmax"
    assert model_sum.hparams.optimizer == "adam"
    assert model_sum.hparams.train_num_ngs == 4
    assert model_sum.hparams.embed_l2 == 0.0
    assert model_sum.hparams.layer_l2 == 0.0
    assert model_sum.hparams.need_sample is True


@pytest.mark.gpu
def test_lightgcn_component_definition(deeprec_config_path):
    yaml_file = os.path.join(deeprec_config_path, "lightgcn.yaml")

    df = movielens.load_pandas_df(size="100k")
    train, test = python_stratified_split(df, ratio=0.75)

    data = ImplicitCF(train=train, test=test)

    hparams = prepare_hparams(yaml_file, embed_size=64)
    model = LightGCN(hparams, data)

    assert model.norm_adj is not None
    assert model.ua_embeddings.shape == [943, 64]
    assert model.ia_embeddings.shape == [1682, 64]
    assert model.u_g_embeddings is not None
    assert model.pos_i_g_embeddings is not None
    assert model.neg_i_g_embeddings is not None
    assert model.batch_ratings is not None
    assert model.loss is not None
    assert model.opt is not None
    assert model.batch_size == 1024
    assert model.epochs == 1000
