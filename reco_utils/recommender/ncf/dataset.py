import random
import numpy as np
import pandas as pd
import warnings

from reco_utils.common.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


class Dataset(object):
    """Dataset class for NCF"""

    def __init__(
        self,
        train,
        test=None,
        n_neg=4,
        n_neg_test=100,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        binary=True,
        seed=42,
    ):
        """Constructor 
        
        Args:
            train (pd.DataFrame): Training data with at least columns (col_user, col_item, col_rating).
            test (pd.DataFrame): Test data with at least columns (col_user, col_item, col_rating). test can be None, 
                if so, we only process the training data.
            n_neg (int): Number of negative samples for training set.
            n_neg_test (int): Number of negative samples for test set.
            col_user (str): User column name.
            col_item (str): Item column name.
            col_rating (str): Rating column name. 
            col_timestamp (str): Timestamp column name.
            binary (bool): If true, set rating > 0 to rating = 1. 
            seed (int): Seed.
        
        """
        # initialize user and item index
        self.user_idx = None
        self.item_idx = None
        # set negative sampling for training and test
        self.n_neg = n_neg
        self.n_neg_test = n_neg_test
        # get col name of user, item and rating
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_timestamp = col_timestamp
        # data preprocessing for training and test data
        self.train, self.test = self._data_processing(train, test, binary)
        # initialize negative sampling for training and test data
        self._init_train_data()
        self._init_test_data()
        # set random seed
        random.seed(seed)

    def _data_processing(self, train, test, binary):
        """Process the dataset to reindex userID and itemID, also set rating as binary feedback

        Args:
            train (pd.DataFrame): Training data with at least columns (col_user, col_item, col_rating). 
            test (pd.DataFrame): Test data with at least columns (col_user, col_item, col_rating)
                    test can be None, if so, we only process the training data.
            binary (bool): If true, set rating>0 to rating = 1.

        Returns:
            list: train and test pd.DataFrame Dataset, which have been reindexed.
        
        """
        # If testing dataset is None
        df = train if test is None else train.append(test)

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

        return self._reindex(train, binary), self._reindex(test, binary)

    def _reindex(self, df, binary):
        """Process dataset to reindex userID and itemID, also set rating as binary feedback

        Args:
            df (pandas.DataFrame): dataframe with at least columns (col_user, col_item, col_rating) 
            binary (bool): if true, set rating>0 to rating = 1 

        Returns:
            list: train and test pandas.DataFrame Dataset, which have been reindexed.
        
        """

        # If testing dataset is None
        if df is None:
            return None

        # Map user_idx and item_idx
        df = pd.merge(df, self.user_idx, on=self.col_user, how="left")
        df = pd.merge(df, self.item_idx, on=self.col_item, how="left")

        # If binary feedback, set rating as 1.0 or 0.0
        if binary:
            df[self.col_rating] = df[self.col_rating].apply(lambda x: float(x > 0))

        # Select relevant columns
        df_reindex = df[
            [self.col_user + "_idx", self.col_item + "_idx", self.col_rating]
        ]
        df_reindex.columns = [self.col_user, self.col_item, self.col_rating]

        return df_reindex

    def _init_train_data(self):
        """Return all negative items (in train dataset) and store them in self.interact_status[self.col_item + '_negative']
        store train dataset in self.users, self.items and self.ratings
        
        """

        self.item_pool = set(self.train[self.col_item].unique())
        self.interact_status = (
            self.train.groupby(self.col_user)[self.col_item]
            .apply(set)
            .reset_index()
            .rename(columns={self.col_item: self.col_item + "_interacted"})
        )
        self.interact_status[self.col_item + "_negative"] = self.interact_status[
            self.col_item + "_interacted"
        ].apply(lambda x: self.item_pool - x)

        self.users, self.items, self.ratings = [], [], []

        # sample n_neg negative samples for training
        for row in self.train.itertuples():
            self.users.append(int(getattr(row, self.col_user)))
            self.items.append(int(getattr(row, self.col_item)))
            self.ratings.append(float(getattr(row, self.col_rating)))

        self.users = np.array(self.users)
        self.items = np.array(self.items)
        self.ratings = np.array(self.ratings)

    def _init_test_data(self):
        """Initialize self.test using 'leave-one-out' evaluation protocol in
            paper https://www.comp.nus.edu.sg/~xiangnan/papers/ncf.pdf
        """
        if self.test is not None:
            # get test positive set for every user
            test_interact_status = (
                self.test.groupby(self.col_user)[self.col_item]
                .apply(set)
                .reset_index()
                .rename(columns={self.col_item: self.col_item + "_interacted_test"})
            )

            # get negative pools for every user based on training and test interactions
            test_interact_status = pd.merge(
                test_interact_status, self.interact_status, on=self.col_user, how="left"
            )
            test_interact_status[
                self.col_item + "_negative"
            ] = test_interact_status.apply(
                lambda row: row[self.col_item + "_negative"]
                - row[self.col_item + "_interacted_test"],
                axis=1,
            )
            test_ratings = pd.merge(
                self.test,
                test_interact_status[[self.col_user, self.col_item + "_negative"]],
                on=self.col_user,
                how="left",
            )

            # sample n_neg_test negative samples for testing
            try:
                test_ratings[self.col_item + "_negative"] = test_ratings[
                    self.col_item + "_negative"
                ].apply(lambda x: random.sample(x, self.n_neg_test))

            except:
                min_num = min(map(len, list(test_ratings[self.col_item + "_negative"])))
                warnings.warn(
                    "n_neg_test is larger than negative items set size! We will set n_neg as the smallest size: %d"
                    % min_num
                )
                test_ratings[self.col_item + "_negative"] = test_ratings[
                    self.col_item + "_negative"
                ].apply(lambda x: random.sample(x, min_num))

            self.test_data = []

            # generate test data
            for row in test_ratings.itertuples():
                self.test_users, self.test_items, self.test_ratings = [], [], []

                self.test_users.append(int(getattr(row, self.col_user)))
                self.test_items.append(int(getattr(row, self.col_item)))
                self.test_ratings.append(float(getattr(row, self.col_rating)))

                for i in getattr(row, self.col_item + "_negative"):
                    self.test_users.append(int(getattr(row, self.col_user)))
                    self.test_items.append(int(i))
                    self.test_ratings.append(float(0))

                self.test_data.append(
                    [
                        [self.id2user[x] for x in self.test_users],
                        [self.id2item[x] for x in self.test_items],
                        self.test_ratings,
                    ]
                )

    def negative_sampling(self):
        """Sample n_neg negative items per positive item, this function should be called every epoch."""
        self.users, self.items, self.ratings = [], [], []

        # sample n_neg negative samples for training
        train_ratings = pd.merge(
            self.train,
            self.interact_status[[self.col_user, self.col_item + "_negative"]],
            on=self.col_user,
        )

        try:
            train_ratings[self.col_item + "_negative"] = train_ratings[
                self.col_item + "_negative"
            ].apply(lambda x: random.sample(x, self.n_neg))
        except:
            min_num = min(map(len, list(train_ratings[self.col_item + "_negative"])))
            warnings.warn(
                "n_neg is larger than negative items set size! We will set n_neg as the smallest size: %d"
                % min_num
            )
            train_ratings[self.col_item + "_negative"] = train_ratings[
                self.col_item + "_negative"
            ].apply(lambda x: random.sample(x, min_num))

        # generate training data
        for row in train_ratings.itertuples():
            self.users.append(int(getattr(row, self.col_user)))
            self.items.append(int(getattr(row, self.col_item)))
            self.ratings.append(float(getattr(row, self.col_rating)))
            for i in getattr(row, self.col_item + "_negative"):
                self.users.append(int(getattr(row, self.col_user)))
                self.items.append(int(i))
                self.ratings.append(float(0))

        self.users = np.array(self.users)
        self.items = np.array(self.items)
        self.ratings = np.array(self.ratings)

    def train_loader(self, batch_size, shuffle=True):
        """Feed train data every batch
        
        Args:
            batch_size (int): Batch size.
            shuffle (bool): Ff true, train data will be shuffled.
        
        Returns:
            list: userID list, itemID list, rating list.
                public data loader return the userID, itemID consistent with raw data

        """

        # yield batch of training data with `shuffle`
        indices = np.arange(len(self.users))
        if shuffle:
            random.shuffle(indices)
        for i in range(len(indices) // batch_size):
            begin_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_indices = indices[begin_idx:end_idx]

            # train_loader() could be called and used by our users in other situations,
            # who expect the not re-indexed data. So we convert id --> original user and item
            # when returning batch

            yield [
                [self.id2user[x] for x in self.users[batch_indices]],
                [self.id2item[x] for x in self.items[batch_indices]],
                self.ratings[batch_indices],
            ]

    def test_loader(self):
        """Feed leave-one-out data every user
        
        Generate test batch by every positive test instance,
        (eg. \[1, 2, 1\] is a positive user & item pair in test set
        (\[userID, itemID, rating\] for this tuple). This function
        returns like \[\[1, 2, 1\], \[1, 3, 0\], \[1,6, 0\], ...\],
        ie. following our *leave-one-out* evaluation protocol.

        Returns:
            list: userID list, itemID list, rating list.
                public data loader return the userID, itemID consistent with raw data
                the first (userID, itemID, rating) is the positive one
        """
        for test in self.test_data:
            yield test
