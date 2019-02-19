# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import tensorflow as tf
import six
import os
from sklearn.metrics import (
    roc_auc_score,
    log_loss,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np
import yaml
import zipfile
from reco_utils.dataset.url_utils import *


def flat_config(config):
    """flat config to a dict"""
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def check_type(config):
    """check config type"""
    # check parameter type
    int_parameters = [
        "word_size",
        "entity_size",
        "doc_size",
        "FEATURE_COUNT",
        "FIELD_COUNT",
        "dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "PAIR_NUM",
        "DNN_FIELD_NUM",
        "attention_layer_sizes",
        "n_user",
        "n_item",
        "n_user_attr",
        "n_item_attr",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("parameters {0} must be int".format(param))

    float_parameters = [
        "init_value",
        "learning_rate",
        "embed_l2",
        "embed_l1",
        "layer_l2",
        "layer_l1",
        "mu",
    ]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("parameters {0} must be float".format(param))

    str_parameters = [
        "train_file",
        "eval_file",
        "test_file",
        "infer_file",
        "method",
        "load_model_name",
        "infer_model_name",
        "loss",
        "optimizer",
        "init_method",
        "attention_activation",
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation", "dropout"]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("parameters {0} must be list".format(param))


def check_nn_config(f_config):
    """check neural networks config"""
    if f_config["model_type"] in ["fm", "FM"]:
        required_parameters = ["FEATURE_COUNT", "dim", "loss", "data_format", "method"]
    elif f_config["model_type"] in ["lr", "LR"]:
        required_parameters = ["FEATURE_COUNT", "loss", "data_format", "method"]
    elif f_config["model_type"] in ["dkn", "DKN"]:
        required_parameters = [
            "doc_size",
            "wordEmb_file",
            "entityEmb_file",
            "word_size",
            "entity_size",
            "data_format",
            "dim",
            "layer_sizes",
            "activation",
            "attention_activation",
            "attention_activation",
            "attention_dropout",
            "loss",
            "data_format",
            "dropout",
            "method",
            "num_filters",
            "filter_sizes",
        ]
    elif f_config["model_type"] in ["exDeepFM", "xDeepFM"]:
        required_parameters = [
            "FIELD_COUNT",
            "FEATURE_COUNT",
            "method",
            "dim",
            "layer_sizes",
            "cross_layer_sizes",
            "activation",
            "loss",
            "data_format",
            "dropout",
        ]
    else:
        required_parameters = [
            "FIELD_COUNT",
            "FEATURE_COUNT",
            "method",
            "dim",
            "layer_sizes",
            "activation",
            "loss",
            "data_format",
            "dropout",
        ]

    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("parameters {0} must be set".format(param))

    if f_config["model_type"] in ["exDeepFM", "xDeepFM"]:
        if f_config["data_format"] != "ffm":
            raise ValueError(
                "for xDeepFM model, data format must be 'ffm', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ["dkn", "DKN"]:
        if f_config["data_format"] != "dkn":
            raise ValueError(
                "for dkn model, data format must be 'dkn', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    else:
        if f_config["data_format"] not in ["fm"]:
            raise ValueError(
                "The default data format should be fm, but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    check_type(f_config)


def check_file_exist(filename):
    if not os.path.isfile(filename):
        raise ValueError("{0} is not exits".format(filename))


def load_yaml_file(filename):
    with open(filename) as f:
        try:
            config = yaml.load(f)
        except:
            raise IOError("load {0} error!".format(filename))
    return config


# train process load yaml
def load_yaml(yaml_name):
    check_file_exist(yaml_name)
    config = load_yaml_file(yaml_name)
    return config


def create_hparams(FLAGS):
    return tf.contrib.training.HParams(
        # data
        kg_file=FLAGS["kg_file"] if "kg_file" in FLAGS else None,
        user_clicks=FLAGS["user_clicks"] if "user_clicks" in FLAGS else None,
        FEATURE_COUNT=FLAGS["FEATURE_COUNT"] if "FEATURE_COUNT" in FLAGS else None,
        FIELD_COUNT=FLAGS["FIELD_COUNT"] if "FIELD_COUNT" in FLAGS else None,
        data_format=FLAGS["data_format"] if "data_format" in FLAGS else None,
        PAIR_NUM=FLAGS["PAIR_NUM"] if "PAIR_NUM" in FLAGS else None,
        DNN_FIELD_NUM=FLAGS["DNN_FIELD_NUM"] if "DNN_FIELD_NUM" in FLAGS else None,
        n_user=FLAGS["n_user"] if "n_user" in FLAGS else None,
        n_item=FLAGS["n_item"] if "n_item" in FLAGS else None,
        n_user_attr=FLAGS["n_user_attr"] if "n_user_attr" in FLAGS else None,
        n_item_attr=FLAGS["n_item_attr"] if "n_item_attr" in FLAGS else None,
        iterator_type=FLAGS["iterator_type"] if "iterator_type" in FLAGS else None,
        SUMMARIES_DIR=FLAGS["SUMMARIES_DIR"] if "SUMMARIES_DIR" in FLAGS else None,
        MODEL_DIR=FLAGS["MODEL_DIR"] if "MODEL_DIR" in FLAGS else None,
        # dkn
        wordEmb_file=FLAGS["wordEmb_file"] if "wordEmb_file" in FLAGS else None,
        entityEmb_file=FLAGS["entityEmb_file"] if "entityEmb_file" in FLAGS else None,
        doc_size=FLAGS["doc_size"] if "doc_size" in FLAGS else None,
        word_size=FLAGS["word_size"] if "word_size" in FLAGS else None,
        entity_size=FLAGS["entity_size"] if "entity_size" in FLAGS else None,
        entity_dim=FLAGS["entity_dim"] if "entity_dim" in FLAGS else None,
        entity_embedding_method=FLAGS["entity_embedding_method"]
        if "entity_embedding_method" in FLAGS
        else None,
        transform=FLAGS["transform"] if "transform" in FLAGS else None,
        train_ratio=FLAGS["train_ratio"] if "train_ratio" in FLAGS else None,
        # model
        dim=FLAGS["dim"] if "dim" in FLAGS else None,
        layer_sizes=FLAGS["layer_sizes"] if "layer_sizes" in FLAGS else None,
        cross_layer_sizes=FLAGS["cross_layer_sizes"]
        if "cross_layer_sizes" in FLAGS
        else None,
        cross_layers=FLAGS["cross_layers"] if "cross_layers" in FLAGS else None,
        activation=FLAGS["activation"] if "activation" in FLAGS else None,
        cross_activation=FLAGS["cross_activation"]
        if "cross_activation" in FLAGS
        else "identity",
        user_dropout=FLAGS["user_dropout"] if "user_dropout" in FLAGS else False,
        dropout=FLAGS["dropout"] if "dropout" in FLAGS else [0.0],
        attention_layer_sizes=FLAGS["attention_layer_sizes"]
        if "attention_layer_sizes" in FLAGS
        else None,
        attention_activation=FLAGS["attention_activation"]
        if "attention_activation" in FLAGS
        else None,
        attention_dropout=FLAGS["attention_dropout"]
        if "attention_dropout" in FLAGS
        else 0.0,
        model_type=FLAGS["model_type"] if "model_type" in FLAGS else None,
        method=FLAGS["method"] if "method" in FLAGS else None,
        load_saved_model=FLAGS["load_saved_model"]
        if "load_saved_model" in FLAGS
        else False,
        load_model_name=FLAGS["load_model_name"]
        if "load_model_name" in FLAGS
        else None,
        filter_sizes=FLAGS["filter_sizes"] if "filter_sizes" in FLAGS else None,
        num_filters=FLAGS["num_filters"] if "num_filters" in FLAGS else None,
        mu=FLAGS["mu"] if "mu" in FLAGS else None,
        fast_CIN_d=FLAGS["fast_CIN_d"] if "fast_CIN_d" in FLAGS else 0,
        use_Linear_part=FLAGS["use_Linear_part"]
        if "use_Linear_part" in FLAGS
        else False,
        use_FM_part=FLAGS["use_FM_part"] if "use_FM_part" in FLAGS else False,
        use_CIN_part=FLAGS["use_CIN_part"] if "use_CIN_part" in FLAGS else False,
        use_DNN_part=FLAGS["use_DNN_part"] if "use_DNN_part" in FLAGS else False,
        # train
        init_method=FLAGS["init_method"] if "init_method" in FLAGS else "tnormal",
        init_value=FLAGS["init_value"] if "init_value" in FLAGS else 0.01,
        embed_l2=FLAGS["embed_l2"] if "embed_l2" in FLAGS else 0.0000,
        embed_l1=FLAGS["embed_l1"] if "embed_l1" in FLAGS else 0.0000,
        layer_l2=FLAGS["layer_l2"] if "layer_l2" in FLAGS else 0.0000,
        layer_l1=FLAGS["layer_l1"] if "layer_l1" in FLAGS else 0.0000,
        cross_l2=FLAGS["cross_l2"] if "cross_l2" in FLAGS else 0.0000,
        cross_l1=FLAGS["cross_l1"] if "cross_l1" in FLAGS else 0.0000,
        reg_kg=FLAGS["reg_kg"] if "reg_kg" in FLAGS else 0.0000,
        learning_rate=FLAGS["learning_rate"] if "learning_rate" in FLAGS else 0.001,
        lr_rs=FLAGS["lr_rs"] if "lr_rs" in FLAGS else 1,
        lr_kg=FLAGS["lr_kg"] if "lr_kg" in FLAGS else 0.5,
        kg_training_interval=FLAGS["kg_training_interval"]
        if "kg_training_interval" in FLAGS
        else 5,
        max_grad_norm=FLAGS["max_grad_norm"] if "max_grad_norm" in FLAGS else 2,
        is_clip_norm=FLAGS["is_clip_norm"] if "is_clip_norm" in FLAGS else 0,
        dtype=FLAGS["dtype"] if "dtype" in FLAGS else 32,
        loss=FLAGS["loss"] if "loss" in FLAGS else None,
        optimizer=FLAGS["optimizer"] if "optimizer" in FLAGS else "adam",
        epochs=FLAGS["epochs"] if "epochs" in FLAGS else 10,
        batch_size=FLAGS["batch_size"] if "batch_size" in FLAGS else 1,
        enable_BN=FLAGS["enable_BN"] if "enable_BN" in FLAGS else False,
        # show info
        show_step=FLAGS["show_step"] if "show_step" in FLAGS else 1,
        save_model=FLAGS["save_model"] if "save_model" in FLAGS else True,
        save_epoch=FLAGS["save_epoch"] if "save_epoch" in FLAGS else 5,
        metrics=FLAGS["metrics"] if "metrics" in FLAGS else None,
        write_tfevents=FLAGS["write_tfevents"] if "write_tfevents" in FLAGS else False,
    )


def prepare_hparams(yaml_file=None, **kwargs):
    if yaml_file:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    if kwargs:
        for name, value in six.iteritems(kwargs):
            config[name] = value

    check_nn_config(config)
    hparams = create_hparams(config)
    return hparams


def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))


def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss"""
    res = {}
    for metric in metrics:
        if metric == "auc":
            auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
            res["auc"] = round(auc, 4)
        elif metric == "rmse":
            rmse = mean_squared_error(np.asarray(labels), np.asarray(preds))
            res["rmse"] = np.sqrt(round(rmse, 4))
        elif metric == "logloss":
            # avoid logloss nan
            preds = [max(min(p, 1.0 - 10e-12), 10e-12) for p in preds]
            logloss = log_loss(np.asarray(labels), np.asarray(preds))
            res["logloss"] = round(logloss, 4)
        elif metric == "acc":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            acc = accuracy_score(np.asarray(labels), pred)
            res["acc"] = round(acc, 4)
        elif metric == "f1":
            pred = np.asarray(preds)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            f1 = f1_score(np.asarray(labels), pred)
            res["f1"] = round(f1, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res
