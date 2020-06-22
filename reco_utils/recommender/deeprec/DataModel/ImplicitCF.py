# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import time
from reco_utils.common.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
)


class ImplicitCF(object):
    """Dataset class for LightGCN"""

    def __init__(
        self,
        train,
        test=None,
        adj_dir=None,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        seed=None,
    ):
        """Constructor 
        
        Args:
            adj_dir (str): Directory to save / load adjacency matrices. If it is None, adjacency
                matrices will be created and will not be saved.
            train (pd.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pd.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                test can be None, if so, we only process the training data.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name. 
            seed (int): Seed.
        
        """
        self.user_idx = None
        self.item_idx = None
        self.adj_dir = adj_dir
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.train, self.test = self._data_processing(train, test)
        self._init_train_data()

        random.seed(seed)

    def _data_processing(self, train, test):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            train (pd.DataFrame): Training data with at least columns (col_user, col_item, col_rating). 
            test (pd.DataFrame): Test data with at least columns (col_user, col_item, col_rating).
                test can be None, if so, we only process the training data.

        Returns:
            list: train and test pd.DataFrame Dataset, which have been reindexed and filtered.
        
        """
        df = train if test is None else train.append(test)

        if self.user_idx is None:
            user_idx = df[[self.col_user]].drop_duplicates().reindex()
            user_idx[self.col_user + "_idx"] = np.arange(len(user_idx))
            self.n_users = len(user_idx)
            self.user_idx = user_idx

            self.user2id = dict(
                zip(user_idx[self.col_user], user_idx[self.col_user + "_idx"])
            )
            self.id2user = {self.user2id[k]: k for k in self.user2id}

        if self.item_idx is None:
            item_idx = df[[self.col_item]].drop_duplicates()
            item_idx[self.col_item + "_idx"] = np.arange(len(item_idx))
            self.n_items = len(item_idx)
            self.item_idx = item_idx

            self.item2id = dict(
                zip(item_idx[self.col_item], item_idx[self.col_item + "_idx"])
            )
            self.id2item = {self.item2id[k]: k for k in self.item2id}

        return self._reindex(train), self._reindex(test)

    def _reindex(self, df):
        """Process the dataset to reindex userID and itemID and only keep records with ratings greater than 0.

        Args:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating).

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed and filtered.
        
        """

        if df is None:
            return None

        df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        df = df[df[self.col_rating] > 0]

        df_reindex = df[
            [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        ]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df_reindex

    def _init_train_data(self):
        """Record items interated with each user in a dataframe self.interact_status, and create adjacency
        matrix self.R.
        
        """
        self.interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )
        self.R = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.R[list(self.train[self.col_user]), list(self.train[self.col_item])] = 1.0

    def get_norm_adj_mat(self):
        """Load normalized adjacency matrix if it exists, otherwise create (and save) it.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.

        """
        try:
            norm_adj_mat = sp.load_npz(self.norm_adj_dir + "/norm_adj_mat.npz")
            print("Already load norm adj matrix.")

        except Exception:
            norm_adj_mat = self.create_norm_adj_mat()
            if self.adj_dir is not None:
                sp.save_npz(self.adj_dir + "/norm_adj_mat.npz", norm_adj_mat)
        return norm_adj_mat

    def create_norm_adj_mat(self):
        """Create normalized adjacency matrix.

        Returns:
            scipy.sparse.csr_matrix: Normalized adjacency matrix.
            
        """
        adj_mat = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()

        adj_mat[: self.n_users, self.n_users :] = R
        adj_mat[self.n_users :, : self.n_users] = R.T
        adj_mat = adj_mat.todok()
        print("Already create adjacency matrix.")

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat_inv = sp.diags(d_inv)
        norm_adj_mat = d_mat_inv.dot(adj_mat)
        norm_adj_mat = norm_adj_mat.dot(d_mat_inv)
        print("Already normalize adjacency matrix.")

        return norm_adj_mat.tocsr()

    def train_loader(self, batch_size):
        """Sample train data every batch. One positive item and one negative item sampled for each user.

        Args:
            batch_size (int): Batch size of users.

        Returns:
            np.array: Sampled users.
            np.array: Sampled positive items.
            np.array: Sampled negative items.

        """

        def sample_neg(x):
            while True:
                neg_id = random.randint(0, self.n_items - 1)
                if neg_id not in x:
                    return neg_id

        indices = range(self.n_users)
        if self.n_users < batch_size:
            users = [random.choice(indices) for _ in range(batch_size)]
        else:
            users = random.sample(indices, batch_size)

        interact = self.interact_status.iloc[users]
        pos_items = interact[self.col_item + "_interacted"].apply(
            lambda x: random.choice(list(x))
        )
        neg_items = interact[self.col_item + "_interacted"].apply(
            lambda x: sample_neg(x)
        )

        return np.array(users), np.array(pos_items), np.array(neg_items)
