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


@pytest.mark.gpu
def test_sasrec_train(base_config, unirec_config_path):
    # config = copy.deepcopy(base_config)
    # yaml_file = os.path.join(unirec_config_path, "model", "SASRec.yaml")
    # config.update(load_yaml(yaml_file))

    # model = SASRec(config)
    import copy
    import datetime
    from recommenders.models.unirec.main import main

    GLOBAL_CONF = {
        # "config_dir": f"{os.path.join(unirec_config_path, 'unirec', 'config')}",
        "config_dir": unirec_config_path,
        "exp_name": "pytest",
        "checkpoint_dir": f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}',
        "model": "",
        "dataloader": "SeqRecDataset",
        "dataset": "",
        "dataset_path": os.path.join(unirec_config_path, "tests/.temp/data"),
        "output_path": "",
        "learning_rate": 0.001,
        "dropout_prob": 0.0,
        "embedding_size": 32,
        "hidden_size": 32,
        "use_pre_item_emb": 0,
        "loss_type": "bce",
        "max_seq_len": 10,
        "has_user_bias": 1,
        "has_item_bias": 1,
        "epochs": 1,
        "early_stop": -1,
        "batch_size": 512,
        "n_sample_neg_train": 9,
        "valid_protocol": "one_vs_all",
        "test_protocol": "one_vs_all",
        "grad_clip_value": 0.1,
        "weight_decay": 1e-6,
        "history_mask_mode": "autoagressive",
        "user_history_filename": "user_history",
        "metrics": "['hit@5;10', 'ndcg@5;10']",
        "key_metric": "ndcg@5",
        "num_workers": 4,
        "num_workers_test": 0,
        "verbose": 2,
        "neg_by_pop_alpha": 0.0,
        "conv_size": 10,  # for ConvFormer-series
    }
    config = copy.deepcopy(GLOBAL_CONF)
    config["task"] = "train"
    config["dataset_path"] = os.path.join(config["dataset_path"], "ml-100k")
    config["dataset"] = "ml-100k"
    config["model"] = "SASRec"
    config["output_path"] = os.path.join(unirec_config_path, f"tests/.temp/output/")
    result = main.run(config)
    assert result is not None
    print(result)
