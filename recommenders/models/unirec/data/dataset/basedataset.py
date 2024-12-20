# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/blob/main/unirec/data/dataset/basedataset.py
#

import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import logging
import torch

# import feather # REMOVED BY MIGUEL (not used in this file)
import copy
from unirec.constants.protocols import *

from unirec.utils.file_io import *


class BaseDataset(Dataset):
    def __init__(self, config, path, filename, transform=None):
        self.config = config
        self.logger = logging.getLogger(config["exp_name"])

        self.logger.info(
            "Constructing dataset of task type: {0}".format(config["data_loader_task"])
        )

        self.dataset_df = self.load_data(path, filename)

        self.use_features = config["use_features"]
        if self.use_features:
            self.features_num = len(eval(config.get("features_shape", "[]")))
            self.item2features = load_features(
                self.config["features_filepath"],
                self.config["n_items"],
                self.features_num,
            )

        ## remove unnessary data columns
        _valid_data_columns = self._get_valid_cols(self.dataset_df.columns.to_list())
        self.dataset_df = self.dataset_df[_valid_data_columns]
        self.data_columns = self.dataset_df.columns

        ## When the format is 'user-item_seq':
        ## if it is training file; or for valid / test, if the eval_protocal is OneVSK;
        ## then transform it to the user-item format
        if self.config["data_format"] in {
            DataFileFormat.T5.value,
            DataFileFormat.T6.value,
        }:
            if self.config["data_loader_task"] == "train" or config[
                "eval_protocol"
            ] in [EvaluationProtocal.OneVSK.value]:
                self.dataset_df = self.expand_dataset(self.dataset_df)
                self.config["data_format"] = DataFileFormat.T1.value

        ## for training dataset, eval_protocol is None
        if config["eval_protocol"] in [
            EvaluationProtocal.OneVSAll.value,
            EvaluationProtocal.OneVSK.value,
        ]:
            if config["data_format"] in [
                DataFileFormat.T2.value,
                DataFileFormat.T2_1.value,
            ]:
                self.logger.info("Remove rows with label == 0")
                _n_rows0 = len(self.dataset_df)
                self.dataset_df = self.dataset_df[
                    self.dataset_df[ColNames.LABEL.value] > 0
                ]
                _n_rows1 = len(self.dataset_df)
                self.logger.info("{0} / {1} rows remains.".format(_n_rows1, _n_rows0))

        if self.config["data_loader_task"] != "train":
            if (
                config["data_format"]
                in [DataFileFormat.T5.value, DataFileFormat.T6.value]
                and config["eval_protocol"] != EvaluationProtocal.OneVSAll.value
            ):
                raise ValueError(
                    "In evaluation, if the data format is T5 or T6, the eval protocal must be OneVSAll."
                )

        if hasattr(self, "_post_process"):
            self._post_process()  # for AERecDataset

        self.dataset = self.dataset_df.values.astype(object)

        del self.dataset_df

        self.transform = transform

        self.set_return_column_index()
        self.logger.info("Finished initializing {0}".format(self.__class__))

    def set_return_column_index(self):
        _type = self.config["data_format"]

        if _type == DataFileFormat.T7.value:
            self.return_key_2_index = {"index_list": 0, "value_list": 1, "label": 2}
        else:
            self.return_key_2_index = {
                "user_id": 0,
                "item_id": 1,
                "label": 2,  ## if the original data format does not contain label, it will append one fake label
            }

        ## additional columns for some dataformat:
        if _type == DataFileFormat.T1_1.value:
            self.return_key_2_index["max_len"] = len(self.return_key_2_index)
        elif _type == DataFileFormat.T2_1.value:
            self.return_key_2_index["session_id"] = len(self.return_key_2_index)
        if self.use_features:
            self.return_key_2_index["item_features"] = len(self.return_key_2_index)

    def expand_dataset(self, dataset_df):
        res = dataset_df[["user_id", "item_seq"]]
        res = res.explode("item_seq", ignore_index=True)
        res = res.rename(columns={"item_seq": "item_id"})
        return res

    def _get_valid_cols(self, candidates):
        _type = self.config["data_format"]
        if _type == DataFileFormat.T1.value:
            t = [
                "user_id",
                "item_id",
            ]
        elif _type == DataFileFormat.T1_1.value:
            t = ["user_id", "item_id", "max_len"]
        elif _type == DataFileFormat.T2.value:
            t = ["user_id", "item_id", "label"]
        elif _type == DataFileFormat.T2_1.value:
            t = ["user_id", "item_id", "label", "session_id"]
        elif _type == DataFileFormat.T3.value:
            t = ["user_id", "item_id", "rating"]
        elif _type == DataFileFormat.T4.value:
            t = ["user_id", "item_id_list", "label_list"]
        elif _type == DataFileFormat.T5.value:
            t = ["user_id", "item_seq"]
        elif _type == DataFileFormat.T6.value:
            t = ["user_id", "item_seq", "time_seq"]
        elif _type == DataFileFormat.T7.value:
            t = ["index_list", "value_list", "label"]
        else:
            raise ValueError(f"The file format `{_type}` is not supported now.")

        candidates = set(candidates)
        res = []
        for a in t:
            if a in candidates:
                res.append(a)
        return res

    r"""
    If there is no label column in the data file,
    it indicates a format of [T1, T3], which will be appended with negative samples
    so only the first item in a group is the original postive item, the rest are all sampled negative.
    """

    def _get_fake_label(self, items):
        if hasattr(self, "fake_label"):
            return self.fake_label
        if isinstance(items, list) or isinstance(items, np.ndarray):
            k = len(items)
            res = np.zeros((k,), dtype=np.int32)
            res[0] = 1
        else:
            res = 1
        self.fake_label = res
        return res

    def _get_label_idx(self):
        _type = self.config["data_format"]
        if _type in [
            DataFileFormat.T2.value,
            DataFileFormat.T2_1.value,
            DataFileFormat.T4.value,
            DataFileFormat.T7.value,
        ]:
            res = 2
        else:
            res = -1
        return res

    def __getitem__(self, index):
        _type = self.config["data_format"]
        sample = self.dataset[index]

        if _type == DataFileFormat.T1_1.value:
            max_len = sample[2]

        if self.transform is not None:
            sample = self.transform(sample)

        if _type == DataFileFormat.T7.value:
            user_id = None
            item_id = None
            index_list = sample[0]
            value_list = sample[1]
        else:
            user_id = sample[0]
            item_id = sample[1]

        if isinstance(item_id, list):
            item_id = np.asarray(item_id)
        # if isinstance(item_id, np.ndarray):#to suppress the warning of pytorch: torch.as_tensor() in default_collate() will change the dataset itself
        #     item_id = copy.deepcopy(item_id)

        label_idx = self._get_label_idx()
        if label_idx >= 0:
            label = sample[label_idx]
        else:
            label = self._get_fake_label(item_id)
        if isinstance(label, list):
            label = np.asarray(label)
        if isinstance(label, np.ndarray) and label_idx >= 0:  # same as item_id
            label = copy.deepcopy(label)

        return_tup = (
            (user_id, item_id, label)
            if user_id is not None
            else (index_list, value_list, label)
        )

        if _type == DataFileFormat.T1_1.value:
            return_tup = return_tup + (max_len,)
        elif _type == DataFileFormat.T2_1.value:
            session_id = sample[3]
            return_tup = return_tup + (session_id,)

        if self.use_features and item_id is not None:
            item_features = self.item2features[
                item_id
            ]  # np.ndarray and int are both supported
            return_tup = return_tup + (item_features,)
        return return_tup

    def __len__(self):
        return len(self.dataset)

    def load_basic_data(self, filename):
        self.logger.debug(
            "loading {0} at {1}".format(
                os.path.basename(filename), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )

        if os.path.exists(filename + ".ftr"):
            data = pd.read_feather(filename + ".ftr")
        elif os.path.exists(filename + ".pkl"):
            data = load_pkl_obj(filename + ".pkl")
        elif (
            os.path.exists(filename + ".tsv")
            or os.path.exists(filename + ".csv")
            or os.path.exists(filename + ".txt")
        ):
            self.logger.info("Loading file : {0} ...".format(filename))
            data = load_txt_file(filename)
            self.logger.info("Done. Data shape is {0}".format(data.shape))
        else:
            raise NotImplementedError("Load plain text data file")

        self.logger.debug(
            "Finished loading {0} at {1}".format(
                os.path.basename(filename), datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            )
        )
        data = data.reset_index(drop=True)

        return data

    def load_data(self, path, filename):
        filename = os.path.join(path, filename)
        data = self.load_basic_data(filename)
        return data
