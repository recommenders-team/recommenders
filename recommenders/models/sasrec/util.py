# Copyright (c) Recommenders contributors.
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
        self.with_time = None

        if self.filename:
            with open(self.filename, "r") as fr:
                sample = fr.readline()
            ncols = len(sample.strip().split(self.col_sep))
            if ncols == 3:
                self.with_time = True
            elif ncols == 2:
                self.with_time = False
            else:
                raise ValueError(f"3 or 2 columns must be in dataset. Given {ncols} columns")

    def split(self, valid_size=1, test_size=1, min_interactions=3, verbose=False):
        """
        Split user interactions into train/valid/test sets using leave-k-out strategy.

        For each user with enough interactions, the last `test_size` items go to test,
        the previous `valid_size` items go to validation, and the rest go to training.
        Users with fewer than `min_interactions` only get training data.

        Args:
            valid_size (int): Number of items per user for validation. Default: 1.
            test_size (int): Number of items per user for testing. Default: 1.
            min_interactions (int): Minimum interactions required for a user to have
                valid/test splits. Users with fewer interactions only get training data.
                Must be >= valid_size + test_size + 1. Default: 3.
            verbose (bool): If True, print split statistics. Default: False.

        Returns:
            dict: Statistics about the split containing:
                - num_users: Total number of users
                - num_items: Total number of items
                - users_with_splits: Number of users with train/valid/test splits
                - users_train_only: Number of users with only training data

        Raises:
            ValueError: If filename is not set or min_interactions is too small.
        """
        if not self.filename:
            raise ValueError("Filename is required. Set it in __init__ or pass as kwarg.")

        # Validate parameters
        required_min = valid_size + test_size + 1
        if min_interactions < required_min:
            raise ValueError(
                f"min_interactions ({min_interactions}) must be >= valid_size + test_size + 1 "
                f"({required_min}) to have at least 1 training item."
            )

        if self.with_time:
            self._data_partition_with_time(valid_size, test_size, min_interactions)
        else:
            self._data_partition(valid_size, test_size, min_interactions)

        # Compute statistics
        users_with_splits = sum(1 for u in self.user_test if len(self.user_test[u]) > 0)
        users_train_only = self.usernum - users_with_splits

        stats = {
            "num_users": self.usernum,
            "num_items": self.itemnum,
            "users_with_splits": users_with_splits,
            "users_train_only": users_train_only,
            "valid_size": valid_size,
            "test_size": test_size,
            "min_interactions": min_interactions,
        }

        if verbose:
            print(f"Split complete:")
            print(f"  Total users: {stats['num_users']}")
            print(f"  Total items: {stats['num_items']}")
            print(f"  Users with train/valid/test: {stats['users_with_splits']}")
            print(f"  Users with train only: {stats['users_train_only']}")
            print(f"  Valid size per user: {valid_size}")
            print(f"  Test size per user: {test_size}")

        return stats

    def _data_partition(self, valid_size, test_size, min_interactions):
        """Internal method to partition data without timestamps."""
        # assume user/item index starting from 1
        with open(self.filename, "r") as f:
            for line in f:
                u, i = line.rstrip().split(self.col_sep)
                u = int(u)
                i = int(i)
                self.usernum = max(u, self.usernum)
                self.itemnum = max(i, self.itemnum)
                self.User[u].append(i)

        for user in self.User:
            nfeedback = len(self.User[user])
            if nfeedback < min_interactions:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                split_point = valid_size + test_size
                self.user_train[user] = self.User[user][:-split_point]
                self.user_valid[user] = self.User[user][-split_point:-test_size] if valid_size > 0 else []
                self.user_test[user] = self.User[user][-test_size:] if test_size > 0 else []

    def _data_partition_with_time(self, valid_size, test_size, min_interactions):
        """Internal method to partition data with timestamps (sorts by time first)."""
        # assume user/item index starting from 1
        with open(self.filename, "r") as f:
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
            if nfeedback < min_interactions:
                self.user_train[user] = self.User[user]
                self.user_valid[user] = []
                self.user_test[user] = []
            else:
                split_point = valid_size + test_size
                self.user_train[user] = self.User[user][:-split_point]
                self.user_valid[user] = self.User[user][-split_point:-test_size] if valid_size > 0 else []
                self.user_test[user] = self.User[user][-test_size:] if test_size > 0 else []
