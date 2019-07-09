# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.recommender.deeprec.deeprec_utils import prepare_hparams, download_deeprec_resources
from reco_utils.recommender.deeprec.models.xDeepFM import XDeepFMModel
from reco_utils.recommender.deeprec.models.dkn import DKN
from reco_utils.recommender.deeprec.IO.iterator import FFMTextIterator
from reco_utils.recommender.deeprec.IO.dkn_iterator import DKNTextIterator


@pytest.fixture
def resource_path():
    return os.path.dirname(os.path.realpath(__file__))


@pytest.mark.gpu
@pytest.mark.deeprec
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
@pytest.mark.deeprec
def test_dkn_component_definition(resource_path):
    data_path = os.path.join(resource_path, "..", "resources", "deeprec", "dkn")
    yaml_file = os.path.join(data_path, "dkn.yaml")
    wordEmb_file = os.path.join(data_path, "word_embeddings_100.npy")
    entityEmb_file = os.path.join(data_path, "TransE_entity2vec_100.npy")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.blob.core.windows.net/deeprec/",
            data_path,
            "dknresources.zip",
        )

    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        entityEmb_file=entityEmb_file,
        epochs=5,
        learning_rate=0.0001,
    )
    assert hparams is not None
    model = DKN(hparams, DKNTextIterator)

    assert model.logit is not None
    assert model.update is not None
    assert model.iterator is not None
