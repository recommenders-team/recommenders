# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/blob/main/unirec/data/dataset/seqrecdataset.py
#

import os
import logging
import time
from datetime import datetime
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import torch
from torch.utils.data import DataLoader, Dataset
from unirec.constants.protocols import DataFileFormat

from unirec.utils.file_io import *
from .basedataset import BaseDataset


class SeqRecDataset(BaseDataset):
    def __init__(self, config, path, filename, transform=None):
        super(SeqRecDataset, self).__init__(config, path, filename, transform)
        self.add_seq_transform = None

    def set_return_column_index(self):
        super(SeqRecDataset, self).set_return_column_index()
        self.return_key_2_index["item_seq"] = len(self.return_key_2_index)
        self.return_key_2_index["item_seq_len"] = len(self.return_key_2_index)

        if self.use_features:
            self.return_key_2_index["item_seq_features"] = len(self.return_key_2_index)
        if self.config["time_seq"]:
            self.return_key_2_index["time_seq"] = len(self.return_key_2_index)

    def add_user_history_transform(self, transform):
        self.add_seq_transform = transform

    def __getitem__(self, index):
        _type = self.config["data_format"]
        elements = super(SeqRecDataset, self).__getitem__(
            index
        )  # user_id, item_id, label, ...
        if _type == DataFileFormat.T1_1.value:
            # elements is (user_id, item_id, label, max_len, ...)
            item_seq, item_seq_len, time_seq = self.add_seq_transform(
                (elements[0], elements[1], elements[3])
            )
        else:
            item_seq, item_seq_len, time_seq = self.add_seq_transform(
                (elements[0], elements[1])
            )
        item_seq = self._padding(item_seq)
        item_seq_len = min(item_seq_len, self.config["max_seq_len"])

        elements = elements + (item_seq, item_seq_len)
        if self.use_features:
            item_seq_features = self.item2features[item_seq]
            elements = elements + (item_seq_features,)
        if self.config["time_seq"]:
            time_seq = self._padding(time_seq)
            elements = elements + (time_seq,)

        return elements

    def _padding(self, x):  # padding item_seq to max_seq_len
        len_seq = len(x)
        k = self.config["max_seq_len"]
        res = np.zeros((k,), dtype=np.int32)
        if len_seq < k:
            res[(k - len_seq) :] = x[:]
        else:
            res[:] = x[len_seq - k :]
        return res
