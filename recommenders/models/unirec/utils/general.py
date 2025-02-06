# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/blob/main/unirec/utils/general.py
#

import time
from typing import List
import numpy as np
import torch
import random
import importlib
import os
import pandas as pd

from .file_io import *
from recommenders.models.unirec.constants.protocols import *


def get_local_time_str():
    return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())


def dict2str(d, sep=" "):
    a = [(k, v) for k, v in d.items()]
    a.sort(key=lambda t: t[0])
    res = sep.join(["{0}:{1}".format(t[0], t[1]) for t in a])
    return res


def init_seed(seed):
    r"""init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True  # could improve efficiency
    torch.backends.cudnn.deterministic = True  # fix random seed


def get_mask(item_seq, emb_size):
    """Generate left-to-right uni-directional attention mask for multi-head attention."""
    mask = (item_seq > 0).long()  # 2048 x 50
    extended_mask = mask.unsqueeze(2)
    extended_mask = extended_mask.expand(item_seq.size()[0], item_seq.size()[1], emb_size)

    return extended_mask


# def get_model(model_name, model_root='model'):
#     r"""Automatically select model class based on model name

#     Args:
#         model_name (str): model name

#     Returns:
#         Recommender: model class
#     """
#     model_file_name = model_name.lower()
#     model_module = None

#     ## the source file path of the model. Subfolder is denoted with `.`. E.g., `model.cf`.
#     ## search the source file from the root folder is `model`.
#     for root, dirs, files in os.walk(model_root, topdown=True):
#         module_path = root.replace('/', '.') + '.' + model_file_name
#         if importlib.util.find_spec(module_path):
#             model_module = importlib.import_module(module_path)
#             break
#     if model_module is None:
#         raise ValueError('Cannot import `model_name` [{0}] from {1}.'.format(model_name, module_path))
#     model_class = getattr(model_module, model_name)
#     return model_class


def get_class_instance(class_name, class_root="model"):
    r"""Automatically select class based on name

    Args:
        class_name (str): class name
        class_root (str): the root directory to search for the target class

    Returns:
        Recommender: a class
    """
    file_name = class_name.lower()
    module = None

    ## the source file path of the model. Subfolder is denoted with `.`. E.g., `model.cf`.
    ## search the source file from the root folder is `model`.

    # the folder of class_root should be infered from the module.
    # When the execution path is not unirec, `os.walk('model')` could not get files.
    class_root = class_root.replace("/", ".")
    submodule_loc = os.path.dirname(importlib.import_module(class_root).__file__)
    for root, dirs, files in os.walk(submodule_loc, topdown=True):
        module_path = root.replace(submodule_loc, class_root).replace("/", ".").replace("\\", ".") + "." + file_name
        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)
            break
    if module is None:
        err_msg = "Cannot import `class name` [{0}] from {1}.".format(class_name, class_root)
        raise ValueError(err_msg)
    target_class = getattr(module, class_name)
    return target_class


r"""
    Load user history from file_name.
    Returns:
        User2History: An N-length ndarray (N is user count) of ndarray (of history items). 
                    E.g., User2History[user_id] is an ndarray of item history for user_id. 
"""


def load_user_history(file_path, file_name, n_users=None, format="user-item", time_seq=0):
    if os.path.exists(os.path.join(file_path, file_name + ".ftr")):
        df = pd.read_feather(os.path.join(file_path, file_name + ".ftr"))
    elif os.path.exists(os.path.join(file_path, file_name + ".pkl")):
        df = load_pkl_obj(os.path.join(file_path, file_name + ".pkl"))
    else:
        raise NotImplementedError("Unsupported user history file type: {0}".format(file_name))

    if format in [DataFileFormat.T1.value, DataFileFormat.T3.value]:
        # ##TODO currently we only support one positive item
        # if isinstance(df['item_id'][0], list) or isinstance(df['item_id'][0], np.ndarray):
        #     df.loc[:, 'item_id'] = df.item_id.apply(lambda x: x[0])
        user_history = df.groupby("user_id")["item_id"].apply(lambda x: np.array(x))
    elif format in [DataFileFormat.T5.value, DataFileFormat.T6.value]:
        # df['item_seq']=df['item_seq'].apply(lambda x:np.array(x))
        user_history = df.set_index("user_id")[ColNames.USER_HISTORY.value].to_dict()
    else:
        raise NotImplementedError("Unsupport user history format: {0}".format(format))

    if n_users is None or n_users <= 0:
        n_users = df["user_id"].max() + 1
        print("Inferred n_users is {0}".format(n_users))
    res = np.empty(n_users, dtype=object)
    for user_id, items in user_history.items():
        res[user_id] = items  # np.fromiter(items, int, len(items))

    res_time = None
    if time_seq:
        if format == DataFileFormat.T3.value:
            user_history_time = df.groupby("user_id")["rating"].apply(lambda x: np.array(x))
        elif format == DataFileFormat.T6.value:
            user_history_time = df.set_index("user_id")[ColNames.TIME_HISTORY.value].to_dict()
        else:
            raise NotImplementedError("Unsupport time history format: {0}".format(format))
        res_time = np.empty(n_users, dtype=object)
        for user_id, times in user_history_time.items():
            res_time[user_id] = times

    return res, res_time


r"""
    Pad a list of np.array to the specific length with value 0.
    Args:
        arrays(List[np.array]): the list of numpy arrays to be padded. List length is batch size and each array in list is a N-D array.
            Currently, we use 1-D array and 2-D array.
            E.g., for 1-D array, the shape is (item_sequence,), and for 2-D array, the shape is (item_sequence, item_seq_features). 
            We will pad the item_sequence dimension to the max_length.
        max_length(int): the specific length. All arrays would be padded to the length. 
                         If None, the max length would be infer from all arrays.
    Returns:
        np.array: the padded array, whose shape is [len(arrays), max_length, *arrays[0].shape[1:]].
"""


def pad_sequence_arrays(arrays: List[np.array], max_length: int = None):
    if max_length is None:
        max_length = max([len(i) for i in arrays])
    shape = (len(arrays), max_length)  # (batch, max_length)
    for i in list(arrays[0].shape[1:]):  # for n-dim array
        shape += (i,)
    res = np.zeros(shape, dtype=arrays[0].dtype)
    for i, array in enumerate(arrays):
        len_array = len(array)
        if len_array < max_length:
            res[i][(max_length - len_array) :] = array[:]
        else:
            res[i][:] = array[(len_array - max_length) :]
    return res


r"""
    Pad a list of np.array to the specific length with value 0.
    Args:
        arrays(List[List[np.array]]): the list of numpy arrays list to be padded. List length is batch size and each array list in list is a list of 1-D array.
            E.g., the array list length is group_size, the shape of array is (n_features, ). 
            We will pad the n_features dimension to the max_length.
        max_length(int): the specific length. All arrays would be padded to the length. 
                         If None, the max length would be infer from all arrays.
    Returns:
        np.array: the padded array, whose shape is [len(arrays), group_size, max_length].
"""


def pad_feature_arrays(arrays: List[List[np.array]], max_length: int = None):
    if max_length is None:
        max_length = max([len(j) for i in arrays for j in i])
    shape = (len(arrays), len(arrays[0]), max_length)  # (batch, group_size, max_length)
    res = np.zeros(shape, dtype=arrays[0][0].dtype)
    for i, array_list in enumerate(arrays):
        for j, array in enumerate(array_list):
            len_array = len(array)
            if len_array < max_length:
                res[i][j][(max_length - len_array) :] = array[:]
            else:
                res[i][j][:] = array[(len_array - max_length) :]
    if len(arrays[0]) == 1:
        res = res.squeeze(1)
    return res


def load_model_freely(filename, device=None):
    if device is not None:
        cpt = torch.load(filename, map_location=device)
    else:
        cpt = torch.load(filename)
    cpt_config = cpt["config"]

    if device is None:
        device = cpt_config["device"]
    else:
        cpt_config["device"] = device

    # remove the path of embedding files, otherwise they will be loaded in the model initialization,
    # which takes time but is useless since they will be overwritten by the state_dict
    if "item_emb_path" in cpt_config:
        del cpt_config["item_emb_path"]
    if "text_emb_path" in cpt_config:
        del cpt_config["text_emb_path"]
    model = get_class_instance(cpt_config["model"], "unirec/model")(cpt_config).to(device)
    model.load_state_dict(cpt["state_dict"], strict=False)
    # model.config['device'] = device
    # model.device = torch.device(device)
    return model, cpt_config


def load_item2info(n_items: int, filename: str, info_type: str):
    r"""Load the information of items from file, e.g. item price and item category.

    Args:
        n_items(int): number of items.
        filename(str): path of the item information file
        info_type(str): the information type, support {'price', 'category'} now

    Return:
        np.ndarray: item information. 1D numpy array, with index being itemid and value being info. Default info for missing itemid is zero.
    """
    assert info_type in {
        "price",
        "category",
    }, "Only support price and category information now."
    print("Load item2{} from {}".format(info_type, filename))
    if filename.endswith(("csv", "tsv")):
        df = pd.read_csv(filename, header=0, sep=",")
    elif filename.endswith("pkl"):
        df = load_pkl_obj(filename)
    elif filename.endswith("ftr"):
        df = pd.read_feather(filename)
    else:
        raise NotImplementedError("Unsupported item {} file type: {}".format(info_type, filename.split("/")[-1]))

    dtype = df[info_type].dtype
    df = df.set_index("item_id")[info_type].to_dict()
    n_items = n_items
    item2info = np.zeros(max(n_items, max(list(df.keys())) + 1), dtype=dtype)
    for item_id, price in df.items():
        item2info[item_id] = price
    return item2info


def get_topk_index(scores, topk):
    r"""Get topk index given scores with numpy. The returned index is sorted by scores descendingly."""
    scores = -scores
    topk_ind = np.argpartition(scores, topk, axis=1)[:, :topk]
    topk_scores = np.take_along_axis(scores, topk_ind, axis=-1)
    sorted_ind_index = np.argsort(topk_scores, axis=1)
    sorted_index = np.take_along_axis(topk_ind, sorted_ind_index, axis=-1)
    return sorted_index
