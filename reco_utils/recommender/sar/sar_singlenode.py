# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Reference implementation of SAR in python/numpy/pandas.

This is not meant to be particularly performant or scalable, just
as a simple and readable implementation.
"""
import numpy as np
import pandas as pd
import logging
from scipy import sparse

from reco_utils.common.python_utils import jaccard, lift, exponential_decay
from reco_utils.evaluation.python_evaluation import get_top_k_items

from reco_utils.common import constants
from reco_utils.recommender import sar


logger = logging.getLogger(__name__)


class SARSingleNode:
    """SAR reference implementation"""

    def __init__(
        self,
        remove_seen=True,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
        col_prediction=constants.PREDICTION_COL,
        similarity_type=sar.SIM_JACCARD,
        time_decay_coefficient=sar.TIME_DECAY_COEFFICIENT,
        time_now=sar.TIME_NOW,
        timedecay_formula=sar.TIMEDECAY_FORMULA,
        threshold=sar.THRESHOLD,
    ):
        """Initialize model parameters

        Args:
            remove_seen (bool): whether to remove items observed in training when making recommendations
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
            similarity_type (str): [None, 'jaccard', 'lift'] option for computing item-item similarity
            time_decay_coefficient (float): number of days till ratings are decayed by 1/2
            time_now (int): current time for time decay calculation
            timedecay_formula (bool): flag to apply time decay
            threshold (int): item-item co-occurrences below this threshold will be removed
        """
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction

        self.remove_seen = remove_seen

        self.similarity_type = similarity_type
        # convert to seconds
        self.time_decay_half_life = time_decay_coefficient * 24 * 60 * 60
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        self.threshold = threshold

        self.model_str = "sar_ref"
        self.user_affinity = None
        self.item_similarity = None

        # threshold - items below this number get set to zero in co-occurrence counts
        assert self.threshold > 0

        # Column for mapping user / item ids to internal indices
        self.col_item_id = sar.INDEXED_ITEMS
        self.col_user_id = sar.INDEXED_USERS

        # Obtain all the users and items from both training and test data
        self.n_users = None
        self.n_items = None

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above map - map array index to actual string ID
        self.index2item = None

        # affinity scores for the recommendation
        self.scores = None

    def compute_affinity_matrix(self, df, n_users, n_items):
        """ Affinity matrix
        The user-affinity matrix can be constructed by treating the users and items as
        indices in a sparse matrix, and the events as the data. Here, we're treating
        the ratings as the event weights.  We convert between different sparse-matrix
        formats to de-duplicate user-item pairs, otherwise they will get added up.
        Args:
            df (pd.DataFrame): Indexed df of users and items.
            n_users (int): Number of users.
            n_items (int): Number of items.
        Returns:
            (sparse.csr): Affinity matrix in Compressed Sparse Row (CSR) format.
        """

        return sparse.coo_matrix(
            (df[self.col_rating], (df[self.col_user_id], df[self.col_item_id])),
            shape=(n_users, n_items),
        ).tocsr()

    def compute_coocurrence_matrix(self, df, n_users, n_items):
        """ Co-occurrence matrix
        C = U'.transpose() * U'
        where U' is the user_affinity matrix with 1's as values (instead of ratings).

        Args:
            df (pd.DataFrame): Indexed df of users and items.
            n_users (int): Number of users.
            n_items (int): Number of items.
        Returns:
            (np.array): Co-occurrence matrix
        """

        user_item_hits = (
            sparse.coo_matrix(
                (
                    np.repeat(1, df.shape[0]),
                    (df[self.col_user_id], df[self.col_item_id]),
                ),
                shape=(n_users, n_items),
            )
            .tocsr()
            .astype(df[self.col_rating].dtype)
        )

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(
            item_cooccurrence >= self.threshold
        )

        return item_cooccurrence

    def fit(self, df):
        """Main fit method for SAR

        Args:
            df (pd.DataFrame): User item rating dataframe
        """

        # Map a continuous index to user / item ids
        self.index2item = dict(enumerate(df[self.col_item].unique()))

        # Invert the mapping from above
        self.item2index = {v: k for k, v in self.index2item.items()}
        self.user2index = {x[1]: x[0] for x in enumerate(df[self.col_user].unique())}

        self.n_users = len(self.user2index)
        self.n_items = len(self.index2item)

        logger.info("Collecting user affinity matrix...")
        assert np.issubdtype(
            df[self.col_rating].dtype, np.floating
        ), "Rating column data type must be floating point"

        # Copy the DataFrame to avoid modification of the input
        temp_df = df[[self.col_user, self.col_item, self.col_rating]].copy()

        if self.time_decay_flag:
            logger.info("Calculating time-decayed affinities...")
            # if time_now is None use the latest time
            if not self.time_now:
                self.time_now = df[self.col_timestamp].max()

            # apply time decay to each rating
            temp_df[self.col_rating] *= exponential_decay(
                value=df[self.col_timestamp],
                max_val=self.time_now,
                half_life=self.time_decay_half_life,
            )

            # group time decayed ratings by user-item and take the sum as the user-item affinity
            temp_df = (
                temp_df.groupby([self.col_user, self.col_item]).sum().reset_index()
            )
        else:
            # without time decay use the latest user-item rating in the dataset as the affinity score
            logger.info("De-duplicating the user-item counts")
            temp_df = temp_df.drop_duplicates(
                [self.col_user, self.col_item], keep="last"
            )

        logger.info("Creating index columns...")
        # Map users and items according to the two dicts. Add the two new columns to newdf.
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].map(self.item2index)
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].map(self.user2index)

        seen_items = None
        if self.remove_seen:
            # retain seen items for removal at prediction time
            seen_items = temp_df[[self.col_user_id, self.col_item_id]].values

        # Affinity matrix
        logger.info("Building user affinity sparse matrix...")
        self.user_affinity = self.compute_affinity_matrix(
            temp_df, self.n_users, self.n_items
        )

        # Calculate item co-occurrence
        logger.info("Calculating item co-occurrence...")
        item_cooccurrence = self.compute_coocurrence_matrix(
            temp_df, self.n_users, self.n_items
        )

        logger.info("Calculating item similarity...")
        if self.similarity_type == sar.SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif self.similarity_type == sar.SIM_JACCARD:
            logger.info("Calculating jaccard ...")
            self.item_similarity = jaccard(item_cooccurrence)
        elif self.similarity_type == sar.SIM_LIFT:
            logger.info("Calculating lift ...")
            self.item_similarity = lift(item_cooccurrence)
        else:
            raise ValueError(
                "Unknown similarity type: {0}".format(self.similarity_type)
            )

        # Calculate raw scores with a matrix multiplication
        logger.info("Calculating recommendation scores...")
        self.scores = self.user_affinity.dot(self.item_similarity)

        # Remove items in the train set so recommended items are always novel
        if self.remove_seen:
            logger.info("Removing seen items...")
            self.scores[seen_items[:, 0], seen_items[:, 1]] = -np.inf

        logger.info("Done training")

    def recommend_k_items(self, test, top_k=10, sort_top_k=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pd.DataFrame): user to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results
        Returns:
            (pd.DataFrame): top k recommendation items for each user
        """

        # get user / item indices from test set
        user_ids = test[self.col_user].drop_duplicates().map(self.user2index).values
        assert not any(
            np.isnan(user_ids)
        ), "SAR cannot score users that are not in the training set"

        # extract only the scores for the test users
        test_scores = self.scores[user_ids, :]

        # ensure we're working with a dense matrix
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.todense()

        # get top K items and scores
        logger.info("Getting top K...")
        # this determines the un-ordered top-k item indices for each user
        top_items = np.argpartition(test_scores, -top_k, axis=1)[:, -top_k:]
        top_scores = test_scores[np.arange(test_scores.shape[0])[:, None], top_items]

        if sort_top_k:
            sort_ind = np.argsort(-top_scores)
            top_items = top_items[np.arange(top_items.shape[0])[:, None], sort_ind]
            top_scores = top_scores[np.arange(top_scores.shape[0])[:, None], sort_ind]

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test[self.col_user].drop_duplicates().values, top_k
                ),
                self.col_item: [self.index2item[item] for item in np.array(top_items).flatten()],
                self.col_prediction: np.array(top_scores).flatten(),
            }
        )

        # ensure datatypes are correct
        df = df.astype(
            dtype={
                self.col_user: str,
                self.col_item: str,
                self.col_prediction: self.scores.dtype,
            }
        )

        # drop seen items
        return df.replace(-np.inf, np.nan).dropna()

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set
        Args:
            test (pd.DataFrame): DataFrame that contains users and items to test
        Return:
            pd.DataFrame: DataFrame contains the prediction results
        """

        # get user / item indices from test set
        user_ids = test[self.col_user].map(self.user2index).values
        assert not any(
            np.isnan(user_ids)
        ), "SAR cannot score users that are not in the training set"

        # extract only the scores for the test users
        test_scores = self.scores[user_ids, :]

        # convert and flatten scores into an array
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.todense()

        item_ids = test[self.col_item].map(self.item2index).values
        nans = np.isnan(item_ids)
        if any(nans):
            # predict 0 for items not seen during training
            test_scores = np.append(test_scores, np.zeros((self.n_users, 1)), axis=1)
            item_ids[nans] = self.n_items
            item_ids = item_ids.astype("int64")

        df = pd.DataFrame(
            {
                self.col_user: test[self.col_user].values,
                self.col_item: test[self.col_item].values,
                self.col_prediction: test_scores[
                    np.arange(test_scores.shape[0]), item_ids
                ],
            }
        )

        # ensure datatypes are correct
        df = df.astype(
            dtype={
                self.col_user: str,
                self.col_item: str,
                self.col_prediction: self.scores.dtype,
            }
        )

        return df
