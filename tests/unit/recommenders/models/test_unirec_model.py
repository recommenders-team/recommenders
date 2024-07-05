# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import copy
import pytest

try:
    import torch
    import torch.nn as nn

    from recommenders.models.unirec.utils.file_io import load_yaml
    from recommenders.models.unirec.model.sequential.seqrec_base import SeqRecBase
    from recommenders.models.unirec.model.sequential.sasrec import SASRec
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.gpu
@pytest.fixture(scope="module")
def base_config(unirec_config_path):
    base_file = os.path.join(unirec_config_path, "base.yaml")
    config = load_yaml(base_file)
    config["exp_name"] = "pytest"
    config["n_users"] = 1
    config["n_items"] = 1
    config["device"] = "cuda"

    return config


@pytest.mark.gpu
def test_seqrecbase_component_definition(base_config):
    model = SeqRecBase(base_config)

    assert model.config["exp_name"] == "pytest"
    assert model.n_users == 1
    assert model.n_items == 1
    assert model.device == "cuda"
    assert model.loss_type == "bce"
    assert model.embedding_size == 32
    # FIXME: Review https://github.com/microsoft/UniRec/pulls/12
    assert model.hidden_size == 32
    assert model.dropout_prob == 0.0
    assert model.use_pre_item_emb == 0
    assert model.use_text_emb == 0
    assert model.text_emb_size == 768
    assert model.init_method == "normal"
    assert model.use_features == 0
    assert model.group_size == -1
    assert model.SCORE_CLIP == -1
    assert model.has_user_bias is False
    assert model.has_item_bias is False
    assert model.tau == 1.0
    assert model.use_features is False

    assert isinstance(model.item_embedding, nn.Embedding)
    # Size: n_items x embedding_size
    assert model.item_embedding.weight.size() == torch.Size([1, 32])

    assert model.annotations == ["AbstractRecommender", "SeqRecBase"]


@pytest.mark.gpu
def test_sasrec_component_definition(base_config, unirec_config_path):
    config = copy.deepcopy(base_config)
    yaml_file = os.path.join(unirec_config_path, "model", "SASRec.yaml")
    config.update(load_yaml(yaml_file))

    model = SASRec(config)

    assert model.n_layers == 2
    assert model.n_heads == 16
    assert model.inner_size == 512
    assert model.hidden_dropout_prob == 0.5
    assert model.attn_dropout_prob == 0.5
    assert model.hidden_act == "swish"
    assert model.layer_norm_eps == 1e-10
    assert model.use_pos_emb is True
    assert model.max_seq_len == 10
    assert model.hidden_size == 32

    assert isinstance(model.position_embedding, nn.Embedding) is True
    # Size: max_seq_len+1 x hidden_size
    assert model.position_embedding.weight.size() == torch.Size([11, 32])
    assert isinstance(model.trm_encoder, nn.Module) is True
    assert isinstance(model.LayerNorm, nn.LayerNorm) is True
    assert isinstance(model.dropout, nn.Dropout) is True
