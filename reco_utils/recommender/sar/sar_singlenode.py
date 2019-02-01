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
        self.seen_items = None

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

        # user2rowID map for prediction method to look up user affinity vectors
        self.user_map_dict = None
        # mapping for item to matrix element
        self.item_map_dict = None

        # the opposite of the above map - map array index to actual string ID
        self.index2user = None
        self.index2item = None

        # affinity scores for the recommendation
        self.scores = None

    def set_index(self, user_map_dict, item_map_dict, index2user, index2item):
        """MVP2 temporary function to set the index of the sparse dataframe.
        In future releases this will be carried out into the data object and index will be provided
        with the data"""

        # original IDs of users and items in a list
        # later as we modify the algorithm these might not be needed (can use dictionary keys
        # instead)
        self.n_users = len(user_map_dict.keys())
        self.n_items = len(item_map_dict.keys())

        # mapping of original IDs to actual matrix elements
        self.user_map_dict = user_map_dict
        self.item_map_dict = item_map_dict

        # reverse mapping of matrix index to an item
        # TODO: we can make this into an array as well
        self.index2user = index2user
        self.index2item = index2item

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

        return (
            sparse.coo_matrix(
                (df[self.col_rating], (df[self.col_user_id], df[self.col_item_id])), shape=(n_users, n_items)
            )
            .todok()
            .tocsr()
        )

    def compute_coocurrence_matrix(self, df, n_users, n_items):
        """ Co-occurrence matrix
        C = U'.transpose() * U'
        where U' is the user_affinity matrix with 1's as values (instead of ratings).

        Args:
            df (pd.DataFrame): Indexed df of users and items.
            n_users (int): Number of users.
            n_items (int): Number of items.
        Returns:
            (np.array): Coocurrence matrix
        """

        user_item_hits = (
            sparse.coo_matrix(
                (np.repeat(1, df.shape[0]), (df[self.col_user_id], df[self.col_item_id])), shape=(n_users, n_items)
            )
            .todok()
            .tocsr()
            .astype(df[self.col_rating].dtype)
        )

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(item_cooccurrence >= self.threshold)

        return item_cooccurrence

    def fit(self, df):
        """Main fit method for SAR

        Args:
            df (pd.DataFrame): User item rating dataframe
        """

        logger.info("Collecting user affinity matrix...")
        assert np.issubdtype(df[self.col_rating].dtype, np.floating), "Rating column data type must be floating point"

        # copy part of the data frame to avoid modification of the input
        temp_df = df[[self.col_user, self.col_item, self.col_rating]].copy()

        if self.time_decay_flag:
            logger.info("Calculating time-decayed affinities...")
            # if time_now is None use the latest time
            if not self.time_now:
                self.time_now = df[self.col_timestamp].max()

            # apply time decay to each rating
            temp_df[self.col_rating] *= exponential_decay(
                value=df[self.col_timestamp], max_val=self.time_now, half_life=self.time_decay_half_life
            )

            # group time decayed ratings by user-item and take the sum as the user-item affinity
            temp_df = temp_df.groupby([self.col_user, self.col_item]).sum().reset_index()

            """
            # experimental implementation of multiprocessing - in practice for smaller datasets this is not needed
            # leaving here in case anyone wants to actually try this
            # to enable, you need:
            #   conda install dill>=0.2.8.1
            #   pip install multiprocess>=0.70.6.1
            # from multiprocess import Pool, cpu_count
            # 
            # multiprocess uses dill for python3 to serialize lambda functions
            #
            # helper function to parallelize the operation on groups
            def applyParallel(dfGrouped, func):
                with Pool(cpu_count()*2) as p:
                    ret_list = p.map(func, [group for name, group in dfGrouped])
                return pd.concat(ret_list)

            from types import MethodType
            grouped.applyParallel = MethodType(applyParallel, grouped)

            # then replace df.apply with df.applyParallel

            Original implementation of groupby and apply - without optimization
            rating_series = grouped.apply(lambda x: np.sum(np.array(x[self.col_rating]) * np.exp(
                -np.log(2.) * (self.time_now - np.array(x[self.col_timestamp])) / (
                    self.time_decay_coefficient * 24. * 3600))))
            """

        else:
            # without time decay use the latest user-item rating in the dataset as the affinity score
            logger.info("De-duplicating the user-item counts")
            temp_df = temp_df.drop_duplicates([self.col_user, self.col_item], keep="last")

        logger.info("Creating index columns...")
        # Map users and items according to the two dicts. Add the two new columns to newdf.
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].map(self.item_map_dict)
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].map(self.user_map_dict)

        if self.remove_seen:
            # retain seen items for removal at prediction time
            self.seen_items = temp_df[[self.col_user_id, self.col_item_id]].values

        # Affinity matrix
        logger.info("Building user affinity sparse matrix...")
        self.user_affinity = self.compute_affinity_matrix(temp_df, self.n_users, self.n_items)

        # Calculate item co-occurrence
        logger.info("Calculating item co-occurrence...")
        item_cooccurrence = self.compute_coocurrence_matrix(temp_df, self.n_users, self.n_items)

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
            raise ValueError("Unknown similarity type: {0}".format(self.similarity_type))

        # Calculate raw scores with a matrix multiplication.
        logger.info("Calculating recommendation scores...")
        self.scores = self.user_affinity.dot(self.item_similarity)

        logger.info("Done training")

    def recommend_k_items(self, test, top_k=10, sort_top_k=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pd.DataFrame): user to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results in descending order
        Returns:
            (pd.DataFrame): top k recommendation items for each user
        """

        # pick users from test set and
        test_users = test[self.col_user].unique()
        try:
            test_users_training_ids = np.array([self.user_map_dict[user] for user in test_users])
        except KeyError():
            msg = "SAR cannot score test set users which are not in the training set"
            logger.error(msg)
            raise ValueError(msg)

        # shorthand
        scores = self.scores

        # Convert to dense, the following operations are easier.
        logger.info("Converting to dense matrix...")
        if isinstance(scores, np.matrixlib.defmatrix.matrix):
            scores_dense = np.array(scores)
        else:
            scores_dense = scores.todense()

        # Mask out items in the train set.  This only makes sense for some
        # problems (where a user wouldn't interact with an item more than once).
        if self.remove_seen:
            logger.info("Removing seen items...")
            scores_dense[self.seen_items[:, 0], self.seen_items[:, 1]] = 0

        # Get top K items and scores.
        logger.info("Getting top K...")
        top_items = np.argpartition(scores_dense, -top_k, axis=1)[:, -top_k:]
        top_scores = scores_dense[np.arange(scores_dense.shape[0])[:, None], top_items]

        logger.info("Select users from the test set")
        top_items = top_items[test_users_training_ids, :]
        top_scores = top_scores[test_users_training_ids, :]

        logger.info("Creating output dataframe...")

        # Convert to np.array (from view) and flatten
        top_items = np.reshape(np.array(top_items), -1)
        top_scores = np.reshape(np.array(top_scores), -1)

        userids = []
        for u in test_users:
            userids.extend([u] * top_k)

        results = pd.DataFrame.from_dict(
            {self.col_user: userids, self.col_item: top_items, self.col_rating: top_scores}
        )

        # remap user and item indices to IDs
        results[self.col_item] = results[self.col_item].map(self.index2item)

        # do final sort
        if sort_top_k:
            results = (
                results.sort_values(by=[self.col_user, self.col_rating], ascending=False)
                .groupby(self.col_user)
                .apply(lambda x: x)
            )

        # format the dataframe in the end to conform to Suprise return type
        logger.info("Formatting output")

        return (
            results[[self.col_user, self.col_item, self.col_rating]]
            .rename(columns={self.col_rating: self.col_prediction})
            .astype({self.col_user: str, self.col_item: str, self.col_prediction: self.scores.dtype})
        )

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set
        Args:
            test (pd.DataFrame): DataFrame that contains users to test
        Return:
            pd.DataFrame: DataFrame contains the prediction results
        """

        # pick users from test set and
        test_users = test[self.col_user].unique()
        try:
            training_ids = np.array([self.user_map_dict[user] for user in test_users])
            assert training_ids is not None
        except KeyError():
            msg = "SAR cannot score test set users which are not in the training set"
            logger.error(msg)
            raise ValueError(msg)

        # shorthand
        scores = self.scores

        # Convert to dense, the following operations are easier.
        logger.info("Converting to dense array ...")
        scores_dense = scores.toarray()

        # take the intersection between train test items and items we actually need
        test_col_hashed_users = test[self.col_user].map(self.user_map_dict)
        test_col_hashed_items = test[self.col_item].map(self.item_map_dict)

        test_index = pd.concat([test_col_hashed_users, test_col_hashed_items], axis=1).values
        aset = set([tuple(x) for x in self.seen_items])
        bset = set([tuple(x) for x in test_index])

        common_index = np.array([x for x in aset & bset])

        # Mask out items in the train set.  This only makes sense for some
        # problems (where a user wouldn't interact with an item more than once).
        if self.remove_seen and len(aset & bset) > 0:
            logger.info("Removing seen items...")
            scores_dense[common_index[:, 0], common_index[:, 1]] = 0

        final_scores = scores_dense[test_index[:, 0], test_index[:, 1]]

        results = pd.DataFrame.from_dict(
            {self.col_user: test_index[:, 0], self.col_item: test_index[:, 1], self.col_rating: final_scores}
        )

        # remap user and item indices to IDs
        results[self.col_user] = results[self.col_user].map(self.index2user)
        results[self.col_item] = results[self.col_item].map(self.index2item)

        # format the dataframe in the end to conform to Suprise return type
        logger.info("Formatting output")

        # modify test to make it compatible with
        return (
            results[[self.col_user, self.col_item, self.col_rating]]
            .rename(columns={self.col_rating: self.col_prediction})
            .astype({self.col_user: str, self.col_item: str, self.col_prediction: self.scores.dtype})
        )
