# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest

try:
    from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
    from recommenders.models.newsrec.newsrec_utils import prepare_hparams, load_yaml
    import tensorflow as tf
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.parametrize(
    "must_exist_attributes", ["wordEmb_file", "wordDict_file", "userDict_file"]
)
@pytest.mark.gpu
def test_prepare_hparams(must_exist_attributes, deeprec_resource_path):
    wordEmb_file = os.path.join(deeprec_resource_path, "mind", "utils", "embedding.npy")
    userDict_file = os.path.join(
        deeprec_resource_path, "mind", "utils", "uid2index.pkl"
    )
    wordDict_file = os.path.join(
        deeprec_resource_path, "mind", "utils", "word_dict.pkl"
    )
    yaml_file = os.path.join(deeprec_resource_path, "mind", "utils", r"nrms.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(deeprec_resource_path, "mind", "utils"),
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
def test_load_yaml_file(deeprec_resource_path):
    yaml_file = os.path.join(deeprec_resource_path, "mind", "utils", r"nrms.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(deeprec_resource_path, "mind", "utils"),
            "MINDdemo_utils.zip",
        )
    config = load_yaml(yaml_file)
    assert config is not None
