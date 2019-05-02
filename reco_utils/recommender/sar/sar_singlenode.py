# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Reference implementation of SAR in python/numpy/pandas.

This is not meant to be particularly performant or scalable, just
a simple and readable implementation.
"""
import numpy as np
import pandas as pd
import logging
from scipy import sparse

from reco_utils.common.python_utils import jaccard, lift, exponential_decay, get_top_k_scored_items
from reco_utils.common import constants
from reco_utils.recommender import sar


logger = logging.getLogger()


class SARSingleNode:
    """SAR reference implementation"""

    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        similarity_type=sar.SIM_JACCARD,
        time_decay_coefficient=sar.TIME_DECAY_COEFFICIENT,
        time_now=sar.TIME_NOW,
        timedecay_formula=sar.TIMEDECAY_FORMULA,
        threshold=sar.THRESHOLD,
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
            similarity_type (str): [None, 'jaccard', 'lift'] option for computing item-item similarity
            time_decay_coefficient (float): number of days till ratings are decayed by 1/2
            time_now (int | None): current time for time decay calculation
            timedecay_formula (bool): flag to apply time decay
            threshold (int): item-item co-occurrences below this threshold will be removed
        """
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction
        self.similarity_type = similarity_type
        # convert to seconds
        self.time_decay_half_life = time_decay_coefficient * 24 * 60 * 60
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        self.threshold = threshold
        self.user_affinity = None
        self.item_similarity = None
        self.item_frequencies = None

        # threshold - items below this number get set to zero in co-occurrence counts
        if self.threshold <= 0:
            raise ValueError("Threshold cannot be < 1")

        # column for mapping user / item ids to internal indices
        self.col_item_id = sar.INDEXED_ITEMS
        self.col_user_id = sar.INDEXED_USERS

        # obtain all the users and items from both training and test data
        self.n_users = None
        self.n_items = None

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above map - map array index to actual string ID
        self.index2item = None

        # track user-item pairs seen during training
        self.seen_items = None

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
            sparse.csr: Affinity matrix in Compressed Sparse Row (CSR) format.
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
            np.array: Co-occurrence matrix
        """

        user_item_hits = sparse.coo_matrix(
            (np.repeat(1, df.shape[0]), (df[self.col_user_id], df[self.col_item_id])),
            shape=(n_users, n_items),
        ).tocsr()

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(
            item_cooccurrence >= self.threshold
        )

        return item_cooccurrence.astype(df[self.col_rating].dtype)

    def set_index(self, df):
        """Generate continuous indices for users and items to reduce memory usage

        Args:
            df (pd.DataFrame): dataframe with user and item ids
        """

        # generate a map of continuous index values to items
        self.index2item = dict(enumerate(df[self.col_item].unique()))

        # invert the mapping from above
        self.item2index = {v: k for k, v in self.index2item.items()}

        # create mapping of users to continuous indices
        self.user2index = {x[1]: x[0] for x in enumerate(df[self.col_user].unique())}

        # set values for the total count of users and items
        self.n_users = len(self.user2index)
        self.n_items = len(self.index2item)

    def fit(self, df):
        """Main fit method for SAR

        Args:
            df (pd.DataFrame): User item rating dataframe
        """

        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        logger.info("Collecting user affinity matrix")
        if not np.issubdtype(df[self.col_rating].dtype, np.number):
            raise TypeError("Rating column data type must be numeric")

        # copy the DataFrame to avoid modification of the input
        temp_df = df[[self.col_user, self.col_item, self.col_rating]].copy()

        if self.time_decay_flag:
            logger.info("Calculating time-decayed affinities")
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

        logger.info("Creating index columns")
        # map users and items according to the two dicts. Add the two new columns to temp_df.
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].map(self.item2index)
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].map(self.user2index)

        # retain seen items for removal at prediction time
        self.seen_items = temp_df[[self.col_user_id, self.col_item_id]].values

        # affinity matrix
        logger.info("Building user affinity sparse matrix")
        self.user_affinity = self.compute_affinity_matrix(
            temp_df, self.n_users, self.n_items
        )

        # calculate item co-occurrence
        logger.info("Calculating item co-occurrence")
        item_cooccurrence = self.compute_coocurrence_matrix(
            temp_df, self.n_users, self.n_items
        )

        # free up some space
        del temp_df

        self.item_frequencies = item_cooccurrence.diagonal()

        logger.info("Calculating item similarity")
        if self.similarity_type == sar.SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif self.similarity_type == sar.SIM_JACCARD:
            logger.info("Calculating jaccard")
            self.item_similarity = jaccard(item_cooccurrence).astype(
                df[self.col_rating].dtype
            )
        elif self.similarity_type == sar.SIM_LIFT:
            logger.info("Calculating lift")
            self.item_similarity = lift(item_cooccurrence).astype(
                df[self.col_rating].dtype
            )
        else:
            raise ValueError(
                "Unknown similarity type: {0}".format(self.similarity_type)
            )

        # free up some space
        del item_cooccurrence

        logger.info("Done training")

    def score(self, test, remove_seen=False):
        """Score all items for test users

        Args:
            test (pd.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            np.ndarray
        """

        # get user / item indices from test set
        user_ids = test[self.col_user].drop_duplicates().map(self.user2index).values
        if any(np.isnan(user_ids)):
            raise ValueError("SAR cannot score users that are not in the training set")

        # calculate raw scores with a matrix multiplication
        logger.info("Calculating recommendation scores")
        # TODO: only compute scores for users in test
        test_scores = self.user_affinity.dot(self.item_similarity)

        # remove items in the train set so recommended items are always novel
        if remove_seen:
            logger.info("Removing seen items")
            test_scores[self.seen_items[:, 0], self.seen_items[:, 1]] = -np.inf

        test_scores = test_scores[user_ids, :]

        # ensure we're working with a dense ndarray
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.toarray()

        return test_scores

    def get_popularity_based_topk(self, top_k=10, sort_top_k=False):
        """Get top K most frequently occurring items across all users

        Args:
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            pd.DataFrame: top k most popular items
        """

        test_scores = np.array([self.item_frequencies])

        logger.info('Getting top K')
        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        return pd.DataFrame(
            {
                self.col_item: [
                    self.index2item[item] for item in top_items.flatten()
                ],
                self.col_prediction: top_scores.flatten(),
            }
        )

    def get_item_based_topk(self, items, top_k=10, sort_top_k=False):
        """Get top K similar items to provided seed items based on similarity metric defined.
        This method will take a set of items and use them to recommend the most similar items to that set
        based on the similarity matrix fit during training.
        This allows recommendations for cold-users (unseen during training), note - the model is not updated.

        The following options are possible based on information provided in the items input:
        1. Single user or seed of items: only item column (ratings are assumed to be 1)
        2. Single user or seed of items w/ ratings: item column and rating column
        3. Separate users or seeds of items: item and user column (user ids are only used to separate item sets)
        4. Separate users or seeds of items with ratings: item, user and rating columns provided

        Args:
            items (pd.DataFrame): DataFrame with item, user (optional), and rating (optional) columns
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            pd.DataFrame: sorted top k recommendation items
        """

        # convert item ids to indices
        item_ids = items[self.col_item].map(self.item2index)

        # if no ratings were provided assume they are all 1
        if self.col_rating in items.columns:
            ratings = items[self.col_rating]
        else:
            ratings = pd.Series(np.ones_like(item_ids))

        # create local map of user ids
        if self.col_user in items.columns:
            test_users = items[self.col_user]
            user2index = {x[1]: x[0] for x in enumerate(items[self.col_user].unique())}
            user_ids = test_users.map(user2index)
        else:
            # if no user column exists assume all entries are for a single user
            test_users = pd.Series(np.zeros_like(item_ids))
            user_ids = test_users
        n_users = user_ids.drop_duplicates().shape[0]

        # generate pseudo user affinity using seed items
        pseudo_affinity = sparse.coo_matrix(
            (ratings, (user_ids, item_ids)), shape=(n_users, self.n_items)
        ).tocsr()

        # calculate raw scores with a matrix multiplication
        test_scores = pseudo_affinity.dot(self.item_similarity)

        # remove items in the seed set so recommended items are novel
        test_scores[user_ids, item_ids] = -np.inf

        top_items, top_scores = get_top_k_scored_items(scores=test_scores, top_k=top_k, sort_top_k=sort_top_k)

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(test_users.drop_duplicates().values, top_items.shape[1]),
                self.col_item: [
                    self.index2item[item] for item in top_items.flatten()
                ],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def recommend_k_items(self, test, top_k=10, sort_top_k=False, remove_seen=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pd.DataFrame): users to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            pd.DataFrame: top k recommendation items for each user
        """

        test_scores = self.score(test, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(scores=test_scores, top_k=top_k, sort_top_k=sort_top_k)

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(test[self.col_user].drop_duplicates().values, top_items.shape[1]),
                self.col_item: [
                    self.index2item[item] for item in top_items.flatten()
                ],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set
        Args:
            test (pd.DataFrame): DataFrame that contains users and items to test

        Returns:
            pd.DataFrame: DataFrame contains the prediction results
        """

        test_scores = self.score(test)
        user_ids = test[self.col_user].map(self.user2index).values

        # create mapping of new items to zeros
        item_ids = test[self.col_item].map(self.item2index).values
        nans = np.isnan(item_ids)
        if any(nans):
            logger.warning(
                "Items found in test not seen during training, new items will have score of 0"
            )
            test_scores = np.append(test_scores, np.zeros((self.n_users, 1)), axis=1)
            item_ids[nans] = self.n_items
            item_ids = item_ids.astype("int64")

        df = pd.DataFrame(
            {
                self.col_user: test[self.col_user].values,
                self.col_item: test[self.col_item].values,
                self.col_prediction: test_scores[user_ids, item_ids],
            }
        )

        return df
