# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pickle
import pytest

try:
    from recommenders.models.deeprec.deeprec_utils import (
        prepare_hparams,
        download_deeprec_resources,
        load_yaml,
        load_dict,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.parametrize(
    "must_exist_attributes", ["FEATURE_COUNT", "data_format", "dim"]
)
@pytest.mark.gpu
def test_prepare_hparams(deeprec_resource_path, must_exist_attributes):
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )
    hparams = prepare_hparams(yaml_file)

    assert hasattr(hparams, must_exist_attributes) is True


@pytest.mark.gpu
def test_load_yaml_file(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")

    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )
    config = load_yaml(yaml_file)

    assert config is not None


class _Exploit:
    def __reduce__(self):
        return (os.system, ("echo pwned",))


@pytest.mark.gpu
def test_load_dict(tmp_path):
    vocab = {"default_mid": 0, "item1": 1, "item2": 2}
    vocab_file = tmp_path / "vocab.pkl"
    with open(vocab_file, "wb") as f:
        pickle.dump(vocab, f)
    assert load_dict(vocab_file) == vocab

    malicious_file = tmp_path / "malicious.pkl"
    with open(malicious_file, "wb") as f:
        pickle.dump(_Exploit(), f)
    with pytest.raises(pickle.UnpicklingError):
        load_dict(malicious_file)
