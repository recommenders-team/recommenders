# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import tensorflow as tf
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams, load_yaml
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources

from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator


@pytest.mark.smoke
@pytest.mark.gpu
def test_news_iterator(mind_resource_path):
    train_news_file = os.path.join(mind_resource_path, "train", r"news.tsv")
    train_behaviors_file = os.path.join(mind_resource_path, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(mind_resource_path, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(mind_resource_path, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", r"nrms.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    train_iterator = MINDIterator(hparams, hparams.npratio)
    test_iterator = MINDIterator(hparams, -1)

    assert train_iterator is not None
    for res in train_iterator.load_data_from_file(
        train_news_file, train_behaviors_file
    ):
        assert isinstance(res, dict)
        assert len(res) == 5
        break

    assert test_iterator is not None
    for res in test_iterator.load_data_from_file(valid_news_file, valid_behaviors_file):
        assert isinstance(res, dict)
        assert len(res) == 5
        break


@pytest.mark.smoke
@pytest.mark.gpu
def test_naml_iterator(mind_resource_path):
    train_news_file = os.path.join(mind_resource_path, "train", r"news.tsv")
    train_behaviors_file = os.path.join(mind_resource_path, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(mind_resource_path, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(mind_resource_path, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding_all.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(mind_resource_path, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(mind_resource_path, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", r"naml.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        vertDict_file=vertDict_file,
        subvertDict_file=subvertDict_file,
        epochs=1,
        batch_size=1024,
    )
    train_iterator = MINDAllIterator(hparams, hparams.npratio)
    test_iterator = MINDAllIterator(hparams, -1)

    assert train_iterator is not None
    for res in train_iterator.load_data_from_file(
        train_news_file, train_behaviors_file
    ):
        assert isinstance(res, dict)
        assert len(res) == 11
        break

    assert test_iterator is not None
    for res in test_iterator.load_data_from_file(valid_news_file, valid_behaviors_file):
        assert isinstance(res, dict)
        assert len(res) == 11
        break
