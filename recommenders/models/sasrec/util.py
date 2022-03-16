# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from collections import defaultdict


class SASRecDataSet:
    """
    A class for creating SASRec specific dataset used during
    train, validation and testing.

    Attributes:
        usernum: integer, total number of users
        itemnum: integer, total number of items
        User: dict, all the users (keys) with items as values
        Items: set of all the items
        user_train: dict, subset of User that are used for training
        user_valid: dict, subset of User that are used for validation
        user_test: dict, subset of User that are used for testing
        col_sep: column separator in the data file
        filename: data filename
    """

    def __init__(self, **kwargs):
        self.usernum = 0
        self.itemnum = 0
        self.User = defaultdict(list)
        self.Items = set()
        self.user_train = {}
        self.user_valid = {}
        self.user_test = {}
        self.col_sep = kwargs.get("col_sep", " ")
        self.filename = kwargs.get("filename", None)

        if self.filename:
            with open(self.filename, "r") as fr:
                sample = fr.readline()
            ncols = sample.strip().split(self.col_sep)
            if ncols == 3:
                self.with_time = True
            else:
                self.with_time = False

    def split(self, **kwargs):
        self.filename = kwargs.get("filename", self.filename)
        if not self.filename:
            raise ValueError("Filename is required")

        if self.with_time:
            self.data_partition_with_time()
        else:
            self.data_partition()

    def data_partition(self):
        # assume user/item index starting from 1
        f = open(self.filename, "r")
        for line in f:
            u, i = line.rstrip().split(self.col_sep)
            u = int(u)
            i = int(i)
            self.usernum = max(u, self.usernum)
            self.itemnum = max(i, self.itemnum)
            self.User[u].append(i)

        for user in self.User:
            nfeedback = len(self.User[user])
            if nfeedback < 3:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                self.user_train[user] = self.User[user][:-2]
                self.user_valid[user] = []
                self.user_valid[user].append(self.User[user][-2])
                self.user_test[user] = []
                self.user_test[user].append(self.User[user][-1])

    def data_partition_with_time(self):
        # assume user/item index starting from 1
        f = open(self.filename, "r")
        for line in f:
            u, i, t = line.rstrip().split(self.col_sep)
            u = int(u)
            i = int(i)
            t = float(t)
            self.usernum = max(u, self.usernum)
            self.itemnum = max(i, self.itemnum)
            self.User[u].append((i, t))
            self.Items.add(i)

        for user in self.User.keys():
            # sort by time
            items = sorted(self.User[user], key=lambda x: x[1])
            # keep only the items
            items = [x[0] for x in items]
            self.User[user] = items
            nfeedback = len(self.User[user])
            if nfeedback < 3:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                self.user_train[user] = self.User[user][:-2]
                self.user_valid[user] = []
                self.user_valid[user].append(self.User[user][-2])
                self.user_test[user] = []
                self.user_test[user].append(self.User[user][-1])
