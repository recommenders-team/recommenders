# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np
import pandas as pd
import warnings
from math import sqrt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from reco_utils.common.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


class RLRMCdataset(object):
    """
    RLRMC dataset implementation. Creates sparse data structures for RLRMC algorithm
    """

    def __init__(
        self,
        train,
        validation=None,
        test=None,
        mean_center=True,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        # seed=42,
    ):
        """Initialize parameters

        Args:
            train (pandas.DataFrame: training data with at least columns (col_user, col_item, col_rating)
            validation (pandas.DataFrame): validation data with at least columns (col_user, col_item, col_rating). validation can be None, if so, we only process the training data
            mean_center (bool): flag to mean center the ratings in train (and validation) data
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
        """
        # initialize user and item index
        self.user_idx = None
        self.item_idx = None

        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        # set random seed
        # random.seed(seed)

        # data preprocessing for training and validation data
        self._data_processing(train, validation, test, mean_center)

    def _data_processing(self, train, validation=None, test=None, mean_center=True):
        """ process the dataset to reindex userID and itemID 
        Args:
            train (pandas.DataFrame): training data with at least columns (col_user, col_item, col_rating) 
            validation (pandas.DataFrame): validation data with at least columns (col_user, col_item, col_rating). validation can be None, if so, we only process the training data
            mean_center (bool): flag to mean center the ratings in train (and validation) data
        Returns:
            list: train and validation pandas.DataFrame Dataset, which have been reindexed.
        
        """
        # Data processing and reindexing code is adopted from https://github.com/Microsoft/Recommenders/blob/master/reco_utils/recommender/ncf/dataset.py
        # If validation dataset is None
        df = train if validation is None else train.append(validation)
        df = df if test is None else df.append(test)

        # Reindex user and item index
        if self.user_idx is None:
            # Map user id
            user_idx = df[[self.col_user]].drop_duplicates().reindex()
            user_idx[self.col_user + "_idx"] = np.arange(len(user_idx))
            self.n_users = len(user_idx)
            self.user_idx = user_idx

            self.user2id = dict(
                zip(user_idx[self.col_user], user_idx[self.col_user + "_idx"])
            )
            self.id2user = {self.user2id[k]: k for k in self.user2id}

        if self.item_idx is None:
            # Map item id
            item_idx = df[[self.col_item]].drop_duplicates()
            item_idx[self.col_item + "_idx"] = np.arange(len(item_idx))
            self.n_items = len(item_idx)
            self.item_idx = item_idx

            self.item2id = dict(
                zip(item_idx[self.col_item], item_idx[self.col_item + "_idx"])
            )
            self.id2item = {self.item2id[k]: k for k in self.item2id}

        df_train = self._reindex(train)

        d = len(user_idx)  # number of rows
        T = len(item_idx)  # number of columns

        rows_train = df_train["userID"].values
        cols_train = df_train["itemID"].values
        entries_omega = df_train["rating"].values
        if mean_center:
            train_mean = np.mean(entries_omega)
        else:
            train_mean = 0.0
        entries_train = entries_omega - train_mean
        self.model_param = {"num_row": d, "num_col": T, "train_mean": train_mean}

        self.train = csr_matrix(
            (entries_train.T.ravel(), (rows_train, cols_train)), shape=(d, T)
        )

        if validation is not None:
            df_validation = self._reindex(validation)
            rows_validation = df_validation["userID"].values
            cols_validation = df_validation["itemID"].values
            entries_validation = df_validation["rating"].values - train_mean
            self.validation = csr_matrix(
                (entries_validation.T.ravel(), (rows_validation, cols_validation)),
                shape=(d, T),
            )
        else:
            self.validation = None

    def _reindex(self, df):
        """ process dataset to reindex userID and itemID
        Args:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating) 
        Returns:
            list: train and validation pandas.DataFrame Dataset, which have been reindexed.
        
        """

        # If validation dataset is None
        if df is None:
            return None

        # Map user_idx and item_idx
        df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        # Select relevant columns
        df_reindex = df[
            [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        ]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df_reindex
