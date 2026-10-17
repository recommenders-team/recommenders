# Copyright (c) Recommenders contributors.
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
    )
    from recommenders.models.deeprec.io.dkn_dataset import (
        DKNDataset,
        DKNItem2ItemDataset,
    )
    from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
except ImportError:
    pass  # disable error while collecting tests for non-gpu environments


@pytest.mark.gpu
def test_dkn_dataset(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    data_file = os.path.join(data_path, "train_mind_demo.txt")
    news_feature_file = os.path.join(data_path, "doc_feature.txt")
    download_deeprec_resources(
        "https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
        data_path,
        "mind-demo.zip",
    )

    dataset = DKNDataset(
        news_feature_file, os.path.join(data_path, "user_history.txt"), 50
    )
    n_instances = 0
    for batch, impression_ids in dataset.load_data_from_file(data_file, 100):
        size = len(impression_ids)
        assert batch["labels"].shape == (size, 1)
        assert batch["candidate_words"].shape == (size, 10)
        assert batch["candidate_entities"].shape == (size, 10)
        assert batch["clicked_words"].shape == (size, 50, 10)
        assert batch["clicked_entities"].shape == (size, 50, 10)
        n_instances += size
    with open(data_file) as rd:
        assert n_instances == sum(1 for _ in rd)

    # doc_list.txt holds 20 news IDs: 4 groups of neg_num + 2 = 5.
    dataset_item2item = DKNItem2ItemDataset(news_feature_file, neg_num=3)
    batches = list(
        dataset_item2item.load_data_from_file(
            os.path.join(data_path, "doc_list.txt"), 3
        )
    )
    assert [batch["words"].shape for batch, _ in batches] == [(3, 5, 10), (1, 5, 10)]
    assert [len(news_ids) for _, news_ids in batches] == [15, 5]


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
