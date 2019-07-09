# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import tensorflow as tf
from reco_utils.recommender.deeprec.deeprec_utils import (
    prepare_hparams,
    download_deeprec_resources,
    load_yaml,
)
from reco_utils.recommender.deeprec.IO.iterator import FFMTextIterator
from reco_utils.recommender.deeprec.IO.dkn_iterator import DKNTextIterator


@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.parametrize(
    "must_exist_attributes", ["FEATURE_COUNT", "data_format", "dim"]
)
@pytest.mark.gpu
@pytest.mark.deeprec
def test_prepare_hparams(must_exist_attributes, resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )
    hparams = prepare_hparams(yaml_file)
    assert hasattr(hparams, must_exist_attributes)


@pytest.mark.gpu
@pytest.mark.deeprec
def test_load_yaml_file(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    config = load_yaml(yaml_file)
    assert config is not None


@pytest.mark.gpu
@pytest.mark.deeprec
def test_FFM_iterator(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    data_file = os.path.join(data_path, "sample_FFM_data.txt")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )

    hparams = prepare_hparams(yaml_file)
    iterator = FFMTextIterator(hparams, tf.Graph())
    assert iterator is not None
    for res in iterator.load_data_from_file(data_file):
        assert isinstance(res, dict)


@pytest.mark.gpu
@pytest.mark.deeprec
def test_DKN_iterator(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "dkn")
    data_file = os.path.join(data_path, "final_test_with_entity.txt")
    yaml_file = os.path.join(data_path, "dkn.yaml")
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "dknresources.zip",
        )

    hparams = prepare_hparams(yaml_file, wordEmb_file="", entityEmb_file="")
    iterator = DKNTextIterator(hparams, tf.Graph())
    assert iterator is not None
    for res in iterator.load_data_from_file(data_file):
        assert isinstance(res, dict)
