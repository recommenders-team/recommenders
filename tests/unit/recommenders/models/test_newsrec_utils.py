# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest

from recommenders.models.newsrec.newsrec_utils import get_mind_data_set, word_tokenize


def test_word_tokenize_lowercases_and_keeps_punctuation():
    assert word_tokenize("Hello, World! It's 2019.") == [
        "hello",
        ",",
        "world",
        "!",
        "it",
        "s",
        "2019",
        ".",
    ]
    assert word_tokenize(float("nan")) == []


def test_get_mind_data_set():
    url, train, dev, utils = get_mind_data_set("demo")
    assert url == "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/"
    assert (train, dev, utils) == (
        "MINDdemo_train.zip",
        "MINDdemo_dev.zip",
        "MINDdemo_utils.zip",
    )
    with pytest.raises(AssertionError):
        get_mind_data_set("tiny")
