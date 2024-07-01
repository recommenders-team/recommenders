# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.

import pickle as pkl
import json
import yaml
import os
from datetime import datetime
import numpy as np
import pandas as pd


## Before Py3.8, pkl default is protocol 3;  otherwise default is 4.
## Protocal=4 supports big object
## So if you are using Python version older than 3.8, you had better pass protocol=4 to this function
def save_pickle(obj, filename, protocol=None):
    print(
        "saving {0} at {1}".format(
            os.path.basename(filename), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )
    with open(filename, "wb") as f:
        if protocol is not None:
            p = pkl.Pickler(f, protocol=protocol)
        else:
            p = pkl.Pickler(f)
        p.fast = True
        p.dump(obj)
    print(
        "finish saving {0} at {1}".format(
            os.path.basename(filename), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )


def load_txt_file(filename):
    sep = None
    if os.path.exists(filename + ".tsv"):
        sep = "\t"
        filename = filename + ".tsv"
    elif os.path.exists(filename + "csv"):
        sep = ","
        filename = filename + ".csv"
    elif os.path.exists(filename + "txt"):
        sep = " "
        filename = filename + ".txt"
    else:
        raise ValueError("Unrecognized filename: {0}".format(filename))

    ## the first line of file must be header
    header = None
    with open(filename, "r") as rd:
        header = rd.readline()
    names = header[:-1].split(sep)
    data = pd.read_csv(filename, sep=sep, header=0, names=names)
    return data


def load_pkl_obj(filename):
    # print('INFO: loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    with open(filename, "rb") as f:
        obj = pkl.load(f)
        # p = pkl.Unpickler(f)
        # obj = p.load()
    # print('finish loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    return obj


def load_npy_obj(filename):
    return np.load(filename)


def save_json(obj, filename):
    with open(filename, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(filename):
    with open(filename, "r") as f:
        obj = json.load(f)
    return obj


def load_yaml(filename):
    with open(filename, "r") as f:
        obj = yaml.load(f, Loader=yaml.FullLoader)
    if obj is None:
        obj = dict()
    return obj


def save_yaml(obj, filename):
    with open(filename, "w") as f:
        yaml.dump(obj, f, indent=4, sort_keys=False)


def _transfer_emb(x):
    x = x.split(",")
    new_x = [float(x_) for x_ in x]
    return new_x


def load_pre_item_emb(file_path, logger):
    logger.info("loading pretrained item embeddings...")

    item_emb_data = pd.read_csv(file_path, names=["id", "emb"], sep="\t")

    item_emb_data["emb"] = item_emb_data["emb"].apply(lambda x: _transfer_emb(x))
    item_emb_ = item_emb_data["emb"].values

    if 0 in item_emb_data["id"].values:
        item_emb_ = item_emb_[1:]

    item_emb = []
    for ie in item_emb_:
        item_emb.append(ie)
    item_emb = np.array(item_emb)

    return item_emb


def load_txt_as_dataframe(infile, sep=",", names=None, dtypes=None):
    print(
        "Loading {0} at {1}".format(
            os.path.basename(infile), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        )
    )

    data = pd.read_csv(
        os.path.join(infile), sep=sep, header=0, names=names, dtype=dtypes
    )

    print(
        "Finish at {0}, data shape is {1}".format(
            datetime.now().strftime("%d/%m/%Y %H:%M:%S"), data.shape
        )
    )
    return data


def load_features(path, n_items, features_num):
    """
    Load features from file.
    The file format is:
        item_id \t feature_1 \t feature_2 \t ... \t feature_n \n
    The first line is header, which is ignored.
    Currently, we only support categorical features.

    Return: features(np.ndarray), shape=(n_items, features_num), each line are features_num "feature fields", and each feature field i has a one-hot feature value: feature_i.
    """
    features = np.zeros((n_items, features_num), dtype=np.int32)
    with open(path, "r") as f:
        _ = f.readline()
        for line in f:
            line = line.strip().split("\t")
            features[int(line[0])] = np.array(line[1:], dtype=np.int32)
    return features


__all__ = [
    "save_pickle",
    "load_txt_file",
    "load_pkl_obj",
    "load_npy_obj",
    "save_json",
    "load_json",
    "load_yaml",
    "save_yaml",
    "load_pre_item_emb",
    "load_txt_as_dataframe",
    "load_features",
]
