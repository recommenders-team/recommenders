# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

try:
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.io.mind_all_iterator import MINDAllIterator
    from recommenders.models.newsrec.io.mind_iterator import MINDIterator
    from recommenders.models.newsrec.newsrec_utils import prepare_hparams
    from recommenders.models.newsrec.models.lstur import LSTURModel
    from recommenders.models.newsrec.models.naml import NAMLModel
    from recommenders.models.newsrec.models.npa import NPAModel
    from recommenders.models.newsrec.models.nrms import NRMSModel
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.gpu
def test_nrms_component_definition(mind_resource_path):
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", r"nrms.yaml")

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
    iterator = MINDIterator
    model = NRMSModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None


@pytest.mark.gpu
def test_naml_component_definition(mind_resource_path):
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding_all.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict_all.pkl")
    vertDict_file = os.path.join(mind_resource_path, "utils", "vert_dict.pkl")
    subvertDict_file = os.path.join(mind_resource_path, "utils", "subvert_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", r"naml.yaml")

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
    )
    iterator = MINDAllIterator
    model = NAMLModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None


@pytest.mark.gpu
def test_npa_component_definition(mind_resource_path):
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", r"npa.yaml")

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
    iterator = MINDIterator
    model = NPAModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None


@pytest.mark.gpu
def test_lstur_component_definition(mind_resource_path):
    wordEmb_file = os.path.join(mind_resource_path, "utils", "embedding.npy")
    userDict_file = os.path.join(mind_resource_path, "utils", "uid2index.pkl")
    wordDict_file = os.path.join(mind_resource_path, "utils", "word_dict.pkl")
    yaml_file = os.path.join(mind_resource_path, "utils", r"lstur.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(mind_resource_path, "mind", "utils"),
            "MINDdemo_utils.zip",
        )
    hparams = prepare_hparams(
        yaml_file,
        wordEmb_file=wordEmb_file,
        wordDict_file=wordDict_file,
        userDict_file=userDict_file,
        epochs=1,
    )
    iterator = MINDIterator
    model = LSTURModel(hparams, iterator)

    assert model.model is not None
    assert model.scorer is not None
    assert model.loss is not None
    assert model.train_optimizer is not None
