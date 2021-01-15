# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import tensorflow as tf
from reco_utils.recommender.newsrec.newsrec_utils import prepare_hparams, load_yaml
from reco_utils.recommender.deeprec.deeprec_utils import download_deeprec_resources

from reco_utils.recommender.newsrec.io.mind_iterator import MINDIterator
from reco_utils.recommender.newsrec.io.mind_all_iterator import MINDAllIterator


@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "must_exist_attributes", ["wordEmb_file", "wordDict_file", "userDict_file"]
)
@pytest.mark.gpu
def test_prepare_hparams(must_exist_attributes, tmp):
    wordEmb_file = os.path.join(tmp, "utils", "embedding.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"nrms.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
            "MINDdemo_utils.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    assert hasattr(hparams, must_exist_attributes)


@pytest.mark.gpu
def test_load_yaml_file(tmp):
    yaml_file = os.path.join(tmp, "utils", r"nrms.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
            "MINDdemo_utils.zip",
        )
    config = load_yaml(yaml_file)
    assert config is not None


@pytest.mark.gpu
def test_news_iterator(tmp):
    train_news_file = os.path.join(tmp, "train", r"news.tsv")
    train_behaviors_file = os.path.join(tmp, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(tmp, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(tmp, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(tmp, "utils", "embedding.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"nrms.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
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


@pytest.mark.gpu
def test_naml_iterator(tmp):
    train_news_file = os.path.join(tmp, "train", r"news.tsv")
    train_behaviors_file = os.path.join(tmp, "train", r"behaviors.tsv")
    valid_news_file = os.path.join(tmp, "valid", r"news.tsv")
    valid_behaviors_file = os.path.join(tmp, "valid", r"behaviors.tsv")
    wordEmb_file = os.path.join(tmp, "utils", "embedding_all.npy")
    userDict_file = os.path.join(tmp, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(tmp, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(tmp, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(tmp, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(tmp, "utils", r"naml.yaml")

    if not os.path.exists(train_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "train"),
            "MINDdemo_train.zip",
        )
    if not os.path.exists(valid_news_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "valid"),
            "MINDdemo_dev.zip",
        )
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(tmp, "utils"),
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
