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
from reco_utils.recommender.deeprec.deeprec_utils import flat_config, load_yaml, load_dict
import json
import pickle as pkl


def check_type(config):
    """Check that the config parameters are the correct type
    
    Args:
        config (dict): Configuration dictionary.

    Raises:
        TypeError: If the parameters are not the correct type.
    """

    int_parameters = [
        "word_size",
        "his_size",
        "doc_size",
        "title_size",
        "body_size",
        "vert_num",
        "subvert_num",
        'npratio',
        'word_emb_dim',
        "attention_hidden_dim",
        "epochs",
        "batch_size",
        "show_step",
        "save_epoch",
        "head_num",
        "head_dim",
        "user_num",
        "filter_num",
        "window_size",
        "gru_unit",
        "user_emb_dim",
        "vert_emb_dim",
        "subvert_emb_dim"
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = [
        "learning_rate",
        "dropout"
    ]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))

    str_parameters = [
        "wordEmb_file",
        "method",
        "loss",
        "optimizer",
        "cnn_activation",
        "dense_activation"
        "type"
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation"]
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
    
    if f_config["model_type"] in ['nrms', 'NRMS']:
        required_parameters = [
            "doc_size",
            "his_size",
            "user_num",
            "wordEmb_file",
            "word_size",
            "npratio",
            "data_format",
            "word_emb_dim",
            # nrms
            "head_num",
            "head_dim",
            # attention
            "attention_hidden_dim",
            "loss",
            "data_format",
            "dropout"
        ]
    
    elif f_config["model_type"] in ['naml', 'NAML']:
        required_parameters = [
            "title_size",
            "body_size",
            "his_size",
            "user_num",
            "vert_num",
            "subvert_num",
            "wordEmb_file",
            "word_size",
            "npratio",
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
            "dropout"
        ]
    elif f_config["model_type"] in ['lstur', 'LSTUR']:
        required_parameters = [
            "doc_size",
            "his_size",
            "user_num",
            "wordEmb_file",
            "word_size",
            "npratio",
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
            "dropout"
        ]
    elif f_config["model_type"] in ['npa', 'NPA']:
        required_parameters = [
            "doc_size",
            "his_size",
            "user_num",
            "wordEmb_file",
            "word_size",
            "npratio",
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
            "dropout"
        ]
    else:
        required_parameters = []

    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

    if f_config["model_type"] in ['nrms', 'NRMS', 'lstur', 'LSTUR']:
        if f_config["data_format"] != "news":
            raise ValueError(
                "For nrms and naml model, data format must be 'news', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    elif f_config["model_type"] in ['naml', 'NAML']:
        if f_config["data_format"] != "naml":
            raise ValueError(
                "For nrms and naml model, data format must be 'naml', but your set is {0}".format(
                    f_config["data_format"]
                )
            )
    
    check_type(f_config)




def create_hparams(flags):
    """Create the model hyperparameters.

    Args:
        flags (dict): Dictionary with the model requirements.

    Returns:
        obj: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return tf.contrib.training.HParams(
        # data
        data_format=flags["data_format"] if "data_format" in flags else None,
        iterator_type=flags["iterator_type"] if "iterator_type" in flags else None,
        # models
        wordEmb_file=flags["wordEmb_file"] if "wordEmb_file" in flags else None,
        doc_size=flags["doc_size"] if "doc_size" in flags else None,
        title_size=flags["title_size"] if "title_size" in flags else None,
        body_size=flags["body_size"] if "body_size" in flags else None,
        word_emb_dim=flags["word_emb_dim"] if "word_emb_dim" in flags else None,
        word_size=flags["word_size"] if "word_size" in flags else None,
        user_num=flags["user_num"] if "user_num" in flags else None,
        vert_num=flags["vert_num"] if "vert_num" in flags else None,
        subvert_num=flags["subvert_num"] if "subvert_num" in flags else None,
        his_size=flags["his_size"] if "his_size" in flags else None,
        npratio=flags["npratio"] if "npratio" in flags else None,
        dropout=flags["dropout"] if "dropout" in flags else 0.0,
        attention_hidden_dim=flags["attention_hidden_dim"] if "attention_hidden_dim" in flags else 200, 
        # nrms
        head_num=flags["head_num"] if "head_num" in flags else 4,
        head_dim=flags["head_dim"] if "head_dim" in flags else 100, 
        # naml
        cnn_activation=flags["cnn_activation"] if "cnn_activation" in flags else None,
        dense_activation=flags["dense_activation"] if "dense_activation" in flags else None,
        filter_num=flags["filter_num"] if "filter_num" in flags else 200,
        window_size=flags["window_size"] if "window_size" in flags else 3,
        vert_emb_dim=flags["vert_emb_dim"] if "vert_emb_dim" in flags else 100,
        subvert_emb_dim=flags["subvert_emb_dim"] if "subvert_emb_dim" in flags else 100,
        # lstur
        gru_unit=flags["gru_unit"] if "gru_unit" in flags else 400,
        type=flags["type"] if "type" in flags else 'ini',
        # npa
        user_emb_dim=flags["user_emb_dim"] if "user_emb_dim" in flags else 50,
        # train
        learning_rate=flags["learning_rate"] if "learning_rate" in flags else 0.001,
        loss=flags["loss"] if "loss" in flags else None,
        optimizer=flags["optimizer"] if "optimizer" in flags else "adam",
        epochs=flags["epochs"] if "epochs" in flags else 10,
        batch_size=flags["batch_size"] if "batch_size" in flags else 1,
        # show info
        show_step=flags["show_step"] if "show_step" in flags else 1,
        metrics=flags["metrics"] if "metrics" in flags else None,
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

    if kwargs:
        for name, value in six.iteritems(kwargs):
            config[name] = value

    check_nn_config(config)
    return create_hparams(config)