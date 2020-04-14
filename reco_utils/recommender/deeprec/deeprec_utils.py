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
from reco_utils.dataset.download_utils import maybe_download
import json
import pickle as pkl


def flat_config(config):
    """Flat config loaded from a yaml file to a flat dict.
    
    Args:
        config (dict): Configuration loaded from a yaml file.

    Returns:
        dict: Configuration dictionary.
    """
    f_config = {}
    category = config.keys()
    for cate in category:
        for key, val in config[cate].items():
            f_config[key] = val
    return f_config


def check_type(config):
    """Check that the config parameters are the correct type
    
    Args:
        config (dict): Configuration dictionary.

    Raises:
        TypeError: If the parameters are not the correct type.
    """

    int_parameters = [
        "word_size",
        "entity_size",
        "doc_size",
        "his_size",
        "title_size",
        "body_size",
        "user_num",
        "vert_num",
        "subvert_num",
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
        "item_embedding_dim",
        "cate_embedding_dim",
        "user_embedding_dim",
        "max_seq_length",
        "hidden_size",
        "T",
        "L",
        "n_v",
        "n_h",
        "min_seq_length",
        "attention_size",
        "train_num_ngs",
        "word_emb_dim",
        "user_emb_dim",
        "vert_emb_dim",
        "subvert_emb_dim",
        "attention_hidden_dim",
        # self-att
        "head_num",
        "head_dim",
        # cnn
        "filter_num",
        "window_size",
        "gru_unit",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

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
            raise TypeError("Parameters {0} must be float".format(param))

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
        "cnn_activation",
        "dense_activation",
        "user_vocab",
        "item_vocab",
        "cate_vocab",
        "wordEmb_file",
        "type",

    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation", "dropout", "att_fcn_layer_sizes"]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))


def check_nn_config(f_config):
    """Check neural networks configuration.
    
    Args:
        f_config (dict): Neural network configuration.
    
    Raises:
        ValueError: If the parameters are not correct.
    """
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
    if f_config["model_type"] in ["gru4rec", "GRU4REC", "GRU4Rec"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "hidden_size",
        ]
    elif f_config["model_type"] in ["caser", "CASER", "Caser"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "user_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "T",
            "L",
            "n_v",
            "n_h",
            "min_seq_length",
        ]
    elif f_config["model_type"] in ["asvd", "ASVD", "a2svd", "A2SVD"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
        ]
    elif f_config["model_type"] in ["slirec", "sli_rec", "SLI_REC", "Sli_rec"]:
        required_parameters = [
            "item_embedding_dim",
            "cate_embedding_dim",
            "max_seq_length",
            "loss",
            "method",
            "user_vocab",
            "item_vocab",
            "cate_vocab",
            "attention_size",
            "hidden_size",
            "att_fcn_layer_sizes",
        ]
    elif f_config["model_type"] in ["nrms", "NRMS"]:
        required_parameters = [
            "doc_size",
            "his_size",
            "user_num",
            "wordEmb_file",
            "word_size",
            "data_format",
            "word_emb_dim",
            # nrms
            "head_num",
            "head_dim",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    elif f_config["model_type"] in ["naml", "NAML"]:
        required_parameters = [
            "title_size",
            "body_size",
            "his_size",
            "user_num",
            "vert_num",
            "subvert_num",
            "wordEmb_file",
            "word_size",
            "data_format",
            "word_emb_dim",
            "vert_emb_dim",
            "subvert_emb_dim",
            # naml
            "filter_num",
            "cnn_activation",
            "window_size",
            "dense_activation",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    elif f_config["model_type"] in ["lstur", "LSTUR"]:
        required_parameters = [
            "doc_size",
            "his_size",
            "user_num",
            "wordEmb_file",
            "word_size",
            "data_format",
            "word_emb_dim",
            # lstur
            "gru_unit",
            "type",
            "filter_num",
            "cnn_activation",
            "window_size",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    elif f_config["model_type"] in ["npa", "NPA"]:
        required_parameters = [
            "doc_size",
            "his_size",
            "user_num",
            "wordEmb_file",
            "word_size",
            "data_format",
            "word_emb_dim",
            # npa
            "user_emb_dim",
            "filter_num",
            "cnn_activation",
            "window_size",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout",
        ]
    else:
        required_parameters = []

    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

    if f_config["model_type"] in ["exDeepFM", "xDeepFM"]:
        if f_config["data_format"] != "ffm":
            raise ValueError(
                "For xDeepFM model, data format must be 'ffm', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ["dkn", "DKN"]:
        if f_config["data_format"] != "dkn":
            raise ValueError(
                "For dkn model, data format must be 'dkn', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ["nrms", "NRMS", "lstur", "LSTUR"]:
        if f_config["data_format"] != "news":
            raise ValueError(
                "For nrms and naml model, data format must be 'news', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ["naml", "NAML"]:
        if f_config["data_format"] != "naml":
            raise ValueError(
                "For nrms and naml model, data format must be 'naml', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    check_type(f_config)


def load_yaml(filename):
    """Load a yaml file.

    Args:
        filename (str): Filename.

    Returns:
        dict: Dictionary.
    """
    try:
        with open(filename, "r") as f:
            config = yaml.load(f, yaml.SafeLoader)
        return config
    except FileNotFoundError:  # for file not found
        raise
    except Exception as e:  # for other exceptions
        raise IOError("load {0} error!".format(filename))


def create_hparams(flags):
    """Create the model hyperparameters.

    Args:
        flags (dict): Dictionary with the model requirements.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return tf.contrib.training.HParams(
        # data
        kg_file=flags.get("kg_file", None),
        user_clicks=flags.get("user_clicks", None),
        FEATURE_COUNT=flags.get("FEATURE_COUNT", None),
        FIELD_COUNT=flags.get("FIELD_COUNT", None),
        data_format=flags.get("data_format", None),
        PAIR_NUM=flags.get("PAIR_NUM", None),
        DNN_FIELD_NUM=flags.get("DNN_FIELD_NUM", None),
        n_user=flags.get("n_user", None),
        n_item=flags.get("n_item", None),
        n_user_attr=flags.get("n_user_attr", None),
        n_item_attr=flags.get("n_item_attr", None),
        iterator_type=flags.get("iterator_type", None),
        SUMMARIES_DIR=flags.get("SUMMARIES_DIR", None),
        MODEL_DIR=flags.get("MODEL_DIR", None),
        # dkn
        wordEmb_file=flags.get("wordEmb_file", None),
        entityEmb_file=flags.get("entityEmb_file", None),
        doc_size=flags.get("doc_size", None),
        word_size=flags.get("word_size", None),
        entity_size=flags.get("entity_size", None),
        entity_dim=flags.get("entity_dim", None),
        entity_embedding_method=flags.get("entity_embedding_method", None),
        transform=flags.get("transform", None),
        train_ratio=flags.get("train_ratio", None),
        # model
        dim=flags.get("dim", None),
        layer_sizes=flags.get("layer_sizes", None),
        cross_layer_sizes=flags.get("cross_layer_sizes", None),
        cross_layers=flags.get("cross_layers", None),
        activation=flags.get("activation", None),
        cross_activation=flags.get("cross_activation", "identity"),
        user_dropout=flags.get("user_dropout", False),
        dropout=flags.get("dropout", [0.0]),
        attention_layer_sizes=flags.get("attention_layer_sizes", None),
        attention_activation=flags.get("attention_activation", None),
        attention_dropout=flags.get("attention_dropout", 0.0),
        model_type=flags.get("model_type", None),
        method=flags.get("method", None),
        load_saved_model=flags.get("load_saved_model", False),
        load_model_name=flags.get("load_model_name", None),
        filter_sizes=flags.get("filter_sizes", None),
        num_filters=flags.get("num_filters", None),
        mu=flags.get("mu", None),
        fast_CIN_d=flags.get("fast_CIN_d", 0),
        use_Linear_part=flags.get("use_Linear_part", False),
        use_FM_part=flags.get("use_FM_part", False),
        use_CIN_part=flags.get("use_CIN_part", False),
        use_DNN_part=flags.get("use_DNN_part", False),
        # train
        init_method=flags.get("init_method", "tnormal"),
        init_value=flags.get("init_value", 0.01),
        embed_l2=flags.get("embed_l2", 0.0000),
        embed_l1=flags.get("embed_l1", 0.0000),
        layer_l2=flags.get("layer_l2", 0.0000),
        layer_l1=flags.get("layer_l1", 0.0000),
        cross_l2=flags.get("cross_l2", 0.0000),
        cross_l1=flags.get("cross_l1", 0.0000),
        reg_kg=flags.get("reg_kg", 0.0000),
        learning_rate=flags.get("learning_rate", 0.001),
        lr_rs=flags.get("lr_rs", 1),
        lr_kg=flags.get("lr_kg", 0.5),
        kg_training_interval=flags.get("kg_training_interval", 5),
        max_grad_norm=flags.get("max_grad_norm", 2),
        is_clip_norm=flags.get("is_clip_norm", 0),
        dtype=flags.get("dtype", 32),
        loss=flags.get("loss", None),
        optimizer=flags.get("optimizer", "adam"),
        epochs=flags.get("epochs", 10),
        batch_size=flags.get("batch_size", 1),
        enable_BN=flags.get("enable_BN", False),
        # show info
        show_step=flags.get("show_step", 1),
        save_model=flags.get("save_model", False),
        save_epoch=flags.get("save_epoch", 5),
        metrics=flags.get("metrics", None),
        write_tfevents=flags.get("write_tfevents", False),
        # sequential
        item_embedding_dim=flags.get("item_embedding_dim", None),
        cate_embedding_dim=flags.get("cate_embedding_dim", None),
        user_embedding_dim=flags.get("user_embedding_dim", None),
        train_num_ngs=flags.get("train_num_ngs", 4),
        need_sample=flags.get("need_sample", True),
        embedding_dropout=flags.get("embedding_dropout", 0.3),
        user_vocab=flags.get("user_vocab", None),
        item_vocab=flags.get("item_vocab", None),
        cate_vocab=flags.get("cate_vocab", None),
        pairwise_metrics=flags.get("pairwise_metrics", None),
        EARLY_STOP=flags.get("EARLY_STOP", 100),
        # gru4rec
        max_seq_length=flags.get("max_seq_length", None),
        hidden_size=flags.get("hidden_size", None),
        # caser,
        L=flags.get("L", None),
        T=flags.get("T", None),
        n_v=flags.get("n_v", None),
        n_h=flags.get("n_h", None),
        min_seq_length=flags.get("min_seq_length", 1),
        # sli_rec
        attention_size=flags.get("attention_size", None),
        att_fcn_layer_sizes=flags.get("att_fcn_layer_sizes", None),
        # naml
        title_size=flags.get("title_size", None),
        body_size=flags.get("body_size", None),
        word_emb_dim=flags.get("word_emb_dim", None),
        user_num=flags.get("user_num", None),
        vert_num=flags.get("vert_num", None),
        subvert_num=flags.get("subvert_num", None),
        his_size=flags.get("his_size", None),
        attention_hidden_dim=flags.get("attention_hidden_dim", 200),
        cnn_activation=flags.get("cnn_activation", None),
        dense_activation=flags.get("dense_activation", None),
        filter_num=flags.get("filter_num", 200),
        window_size=flags.get("window_size", 3),
        vert_emb_dim=flags.get("vert_emb_dim", 100),
        subvert_emb_dim=flags.get("subvert_emb_dim", 100),
        # nrms
        head_num=flags.get("head_num", 4),
        head_dim=flags.get("head_dim", 100),
        # lstur
        gru_unit=flags.get("gru_unit", 400),
        type=flags.get("type", "ini"),
        # npa
        user_emb_dim=flags.get("user_emb_dim", 50),
    )


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    config.update(kwargs)

    check_nn_config(config)
    return create_hparams(config)


def download_deeprec_resources(azure_container_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        azure_container_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = azure_container_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))


def mrr_score(y_true, y_score):
    """Computing mrr score metric.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.
    
    Returns:
        numpy.ndarray: mrr scores.
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def ndcg_score(y_true, y_score, k=10):
    """Computing ndcg score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: ndcg scores.
    """
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def hit_score(y_true, y_score, k=10):
    """Computing hit score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: hit score.
    """
    ground_truth = np.where(y_true == 1)[0]
    argsort = np.argsort(y_score)[::-1][:k]
    for idx in argsort:
        if idx in ground_truth:
            return 1
    return 0

def dcg_score(y_true, y_score, k=10):
    """Computing dcg score metric at k.

    Args:
        y_true (numpy.ndarray): ground-truth labels.
        y_score (numpy.ndarray): predicted labels.

    Returns:
        numpy.ndarray: dcg scores.
    """
    k = min(np.shape(y_true)[-1], k)
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)



def cal_metric(labels, preds, metrics):
    """Calculate metrics,such as auc, logloss.
    
    FIXME: 
        refactor this with the reco metrics and make it explicit.
    """
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
        elif metric == "mean_mrr":
            mean_mrr = np.mean(
                [   
                    mrr_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["mean_mrr"] = round(mean_mrr, 4)
        elif metric.startswith("ndcg"):  # format like:  ndcg@2;4;6;8
            ndcg_list = [1, 2]
            ks = metric.split('@')
            if len(ks) > 1:
                ndcg_list = [int(token) for token in ks[1].split(';')]
            for k in ndcg_list:
                ndcg_temp = np.mean(
                    [   
                        ndcg_score(each_labels, each_preds, k)
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["ndcg@{0}".format(k)] = round(ndcg_temp, 4)
        elif metric.startswith("hit"):  # format like:  hit@2;4;6;8
            hit_list = [1, 2]
            ks = metric.split('@')
            if len(ks) > 1:
                hit_list = [int(token) for token in ks[1].split(';')]
            for k in hit_list:
                hit_temp = np.mean(
                    [
                        hit_score(each_labels, each_preds, k) 
                        for each_labels, each_preds in zip(labels, preds)
                    ]
                )
                res["hit@{0}".format(k)] = round(hit_temp, 4)
        elif metric == "group_auc":
            group_auc = np.mean(
                [
                    roc_auc_score(each_labels, each_preds)
                    for each_labels, each_preds in zip(labels, preds)
                ]
            )
            res["group_auc"] = round(group_auc, 4)
        else:
            raise ValueError("not define this metric {0}".format(metric))
    return res


def load_dict(filename):
    """Load the vocabularies.

    Args:
        filename (str): Filename of user, item or category vocabulary.

    Returns:
        dict: A saved vocabulary.
    """
    with open(filename, "rb") as f:
        f_pkl = pkl.load(f)
        return f_pkl
