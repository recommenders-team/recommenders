# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

try:
    import tensorflow as tf
    from recommenders.datasets.amazon_reviews import (
        download_and_extract,
        data_preprocessing,
    )
    from recommenders.models.deeprec.deeprec_utils import (
        prepare_hparams,
        download_deeprec_resources,
        load_yaml,
    )
    from recommenders.models.deeprec.io.dkn_iterator import DKNTextIterator
    from recommenders.models.deeprec.io.dkn_item2item_iterator import (
        DKNItem2itemTextIterator,
    )
    from recommenders.models.deeprec.io.iterator import FFMTextIterator
    from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
    from recommenders.models.deeprec.models.sequential.sli_rec import SLI_RECModel
except ImportError:
    pass  # disable error while collecting tests for non-gpu environments


@pytest.mark.smoke
@pytest.mark.gpu
def test_DKN_iterator(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    data_file = os.path.join(data_path, r"train_mind_demo.txt")
    news_feature_file = os.path.join(data_path, r"doc_feature.txt")
    user_history_file = os.path.join(data_path, r"user_history.txt")
    wordEmb_file = os.path.join(data_path, "word_embeddings_100.npy")
    entityEmb_file = os.path.join(data_path, "TransE_entity2vec_100.npy")
    contextEmb_file = os.path.join(data_path, "TransE_context2vec_100.npy")
    yaml_file = os.path.join(data_path, "dkn.yaml")
    download_deeprec_resources(
        "https://recodatasets.z20.web.core.windows.net/deeprec/",
        data_path,
        "mind-demo.zip",
    )

    hparams = prepare_hparams(
        yaml_file,
        news_feature_file=news_feature_file,
        user_history_file=user_history_file,
        wordEmb_file="",
        entityEmb_file="",
        contextEmb_file="",
    )
    iterator = DKNTextIterator(hparams, tf.Graph())
    assert iterator is not None
    for res, impression, data_size in iterator.load_data_from_file(data_file):
        assert isinstance(res, dict)

    ###  test DKN item2item iterator
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

    iterator_item2item = DKNItem2itemTextIterator(hparams, tf.Graph())
    assert iterator_item2item is not None
    test_round = 3
    for res, impression, data_size in iterator_item2item.load_data_from_file(
        os.path.join(data_path, "doc_list.txt")
    ):
        assert isinstance(res, dict)
        test_round -= 1
        if test_round <= 0:
            break


@pytest.mark.smoke
@pytest.mark.gpu
def test_Sequential_Iterator(deeprec_resource_path, deeprec_config_path):
    data_path = os.path.join(deeprec_resource_path, "slirec")
    yaml_file = os.path.join(deeprec_config_path, "sli_rec.yaml")
    train_file = os.path.join(data_path, r"train_data")

    if not os.path.exists(train_file):
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

    hparams = prepare_hparams(yaml_file)
    iterator = SequentialIterator(hparams, tf.Graph())
    assert iterator is not None
    for res in iterator.load_data_from_file(train_file):
        assert isinstance(res, dict)
