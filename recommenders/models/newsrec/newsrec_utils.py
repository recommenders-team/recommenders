# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import random
import re
import tensorflow as tf

from recommenders.models.deeprec.deeprec_utils import (
    flat_config,
    load_yaml,
)


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
        "title_size",
        "body_size",
        "npratio",
        "word_emb_dim",
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
        "subvert_emb_dim",
    ]
    for param in int_parameters:
        if param in config and not isinstance(config[param], int):
            raise TypeError("Parameters {0} must be int".format(param))

    float_parameters = ["learning_rate", "dropout"]
    for param in float_parameters:
        if param in config and not isinstance(config[param], float):
            raise TypeError("Parameters {0} must be float".format(param))

    str_parameters = [
        "wordEmb_file",
        "wordDict_file",
        "userDict_file",
        "vertDict_file",
        "subvertDict_file",
        "method",
        "loss",
        "optimizer",
        "cnn_activation",
        "dense_activation" "type",
    ]
    for param in str_parameters:
        if param in config and not isinstance(config[param], str):
            raise TypeError("Parameters {0} must be str".format(param))

    list_parameters = ["layer_sizes", "activation"]
    for param in list_parameters:
        if param in config and not isinstance(config[param], list):
            raise TypeError("Parameters {0} must be list".format(param))

    bool_parameters = ["support_quick_scoring"]
    for param in bool_parameters:
        if param in config and not isinstance(config[param], bool):
            raise TypeError("Parameters {0} must be bool".format(param))


def check_nn_config(f_config):
    """Check neural networks configuration.

    Args:
        f_config (dict): Neural network configuration.

    Raises:
        ValueError: If the parameters are not correct.
    """

    if f_config["model_type"] in ["nrms", "NRMS"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
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
            "dropout",
        ]

    elif f_config["model_type"] in ["naml", "NAML"]:
        required_parameters = [
            "title_size",
            "body_size",
            "his_size",
            "wordEmb_file",
            "subvertDict_file",
            "vertDict_file",
            "wordDict_file",
            "userDict_file",
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
            "dropout",
        ]
    elif f_config["model_type"] in ["lstur", "LSTUR"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
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
            "dropout",
        ]
    elif f_config["model_type"] in ["npa", "NPA"]:
        required_parameters = [
            "title_size",
            "his_size",
            "wordEmb_file",
            "wordDict_file",
            "userDict_file",
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
            "dropout",
        ]
    else:
        required_parameters = []

    # check required parameters
    for param in required_parameters:
        if param not in f_config:
            raise ValueError("Parameters {0} must be set".format(param))

    if f_config["model_type"] in ["nrms", "NRMS", "lstur", "LSTUR"]:
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


def create_hparams(flags):
    """Create the model hyperparameters.

    Args:
        flags (dict): Dictionary with the model requirements.

    Returns:
        object: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    return tf.contrib.training.HParams(
        # data
        data_format=flags.get("data_format", None),
        iterator_type=flags.get("iterator_type", None),
        support_quick_scoring=flags.get("support_quick_scoring", False),
        wordEmb_file=flags.get("wordEmb_file", None),
        wordDict_file=flags.get("wordDict_file", None),
        userDict_file=flags.get("userDict_file", None),
        vertDict_file=flags.get("vertDict_file", None),
        subvertDict_file=flags.get("subvertDict_file", None),
        # models
        title_size=flags.get("title_size", None),
        body_size=flags.get("body_size", None),
        word_emb_dim=flags.get("word_emb_dim", None),
        word_size=flags.get("word_size", None),
        user_num=flags.get("user_num", None),
        vert_num=flags.get("vert_num", None),
        subvert_num=flags.get("subvert_num", None),
        his_size=flags.get("his_size", None),
        npratio=flags.get("npratio"),
        dropout=flags.get("dropout", 0.0),
        attention_hidden_dim=flags.get("attention_hidden_dim", 200),
        # nrms
        head_num=flags.get("head_num", 4),
        head_dim=flags.get("head_dim", 100),
        # naml
        cnn_activation=flags.get("cnn_activation", None),
        dense_activation=flags.get("dense_activation", None),
        filter_num=flags.get("filter_num", 200),
        window_size=flags.get("window_size", 3),
        vert_emb_dim=flags.get("vert_emb_dim", 100),
        subvert_emb_dim=flags.get("subvert_emb_dim", 100),
        # lstur
        gru_unit=flags.get("gru_unit", 400),
        type=flags.get("type", "ini"),
        # npa
        user_emb_dim=flags.get("user_emb_dim", 50),
        # train
        learning_rate=flags.get("learning_rate", 0.001),
        loss=flags.get("loss", None),
        optimizer=flags.get("optimizer", "adam"),
        epochs=flags.get("epochs", 10),
        batch_size=flags.get("batch_size", 1),
        # show info
        show_step=flags.get("show_step", 1),
        metrics=flags.get("metrics", None),
    )


def prepare_hparams(yaml_file=None, **kwargs):
    """Prepare the model hyperparameters and check that all have the correct value.

    Args:
        yaml_file (str): YAML file as configuration.

    Returns:
        object: Hyperparameter object in TF (tf.contrib.training.HParams).
    """
    if yaml_file is not None:
        config = load_yaml(yaml_file)
        config = flat_config(config)
    else:
        config = {}

    config.update(kwargs)

    check_nn_config(config)
    return create_hparams(config)


def word_tokenize(sent):
    """Split sentence into word list using regex.
    Args:
        sent (str): Input sentence

    Return:
        list: word list
    """
    pat = re.compile(r"[\w]+|[.,!?;|]")
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []


def newsample(news, ratio):
    """Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def get_mind_data_set(type):
    """Get MIND dataset address

    Args:
        type (str): type of mind dataset, must be in ['large', 'small', 'demo']

    Returns:
        list: data url and train valid dataset name
    """
    assert type in ["large", "small", "demo"]

    if type == "large":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDlarge_train.zip",
            "MINDlarge_dev.zip",
            "MINDlarge_utils.zip",
        )

    elif type == "small":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDsmall_train.zip",
            "MINDsmall_dev.zip",
            "MINDsmall_utils.zip",
        )

    elif type == "demo":
        return (
            "https://recodatasets.z20.web.core.windows.net/newsrec/",
            "MINDdemo_train.zip",
            "MINDdemo_dev.zip",
            "MINDdemo_utils.zip",
        )
