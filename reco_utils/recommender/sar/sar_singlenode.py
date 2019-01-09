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

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
    PREDICTION_COL,
)

from reco_utils.recommender.sar import (
    SIM_JACCARD,
    SIM_LIFT,
    SIM_COOCCUR,
    HASHED_USERS,
    HASHED_ITEMS,
)
from reco_utils.recommender.sar import (
    TIME_DECAY_COEFFICIENT,
    TIME_NOW,
    TIMEDECAY_FORMULA,
    THRESHOLD,
)

"""
enable or set manually with --log=INFO when running example file if you want logging:
disabling because logging output contaminates stdout output on Databricsk Spark clusters
"""
# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SARSingleNode:
    """SAR reference implementation"""

    def __init__(
        self,
        remove_seen=True,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=DEFAULT_TIMESTAMP_COL,
        similarity_type=SIM_JACCARD,
        time_decay_coefficient=TIME_DECAY_COEFFICIENT,
        time_now=TIME_NOW,
        timedecay_formula=TIMEDECAY_FORMULA,
        threshold=THRESHOLD,
        debug=False,
    ):

        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        # default values for all SAR algos
        self.col_timestamp = col_timestamp

        self.remove_seen = remove_seen

        # time of item-item similarity
        self.similarity_type = similarity_type
        # denominator in time decay. Zero makes time decay irrelevant
        self.time_decay_coefficient = time_decay_coefficient
        # toggle the computation of time decay group by formula
        self.timedecay_formula = timedecay_formula
        # current time for time decay calculation
        self.time_now = time_now
        # cooccurrence matrix threshold
        self.threshold = threshold
        # debug the code
        self.debug = debug
        # log the length of operations
        self.timer_log = []

        # array of indexes for rows and columns of users and items in training set
        self.index = None
        self.model_str = "sar_ref"
        self.model = self

        # threshold - items below this number get set to zero in coocurrence counts
        assert self.threshold > 0

        # more columns which are used internally
        self._col_hashed_items = HASHED_ITEMS
        self._col_hashed_users = HASHED_USERS

        # Obtain all the users and items from both training and test data
        self.unique_users = None
        self.unique_items = None
        # store training set index for future use during prediction
        self.index = None

        # user2rowID map for prediction method to look up user affinity vectors
        self.user_map_dict = None
        # mapping for item to matrix element
        self.item_map_dict = None

        # the opposite of the above map - map array index to actual string ID
        self.index2user = None
        self.index2item = None

        # affinity scores for the recommendation
        self.scores = None

    def set_index(
        self,
        unique_users,
        unique_items,
        user_map_dict,
        item_map_dict,
        index2user,
        index2item,
    ):
        """MVP2 temporary function to set the index of the sparse dataframe.
        In future releases this will be carried out into the data object and index will be provided
        with the data"""

        # original IDs of users and items in a list
        # later as we modify the algorithm these might not be needed (can use dictionary keys
        # instead)
        self.unique_users = unique_users
        self.unique_items = unique_items

        # mapping of original IDs to actual matrix elements
        self.user_map_dict = user_map_dict
        self.item_map_dict = item_map_dict

        # reverse mapping of matrix index to an item
        # TODO: we can make this into an array as well
        self.index2user = index2user
        self.index2item = index2item

    # private methods
    @staticmethod
    def __jaccard(cooccurrence):
        """Helper method to calculate teh Jaccard cooccurrence of the item-item similarity"""
        log.info("Calculating jaccard...")
        diag = cooccurrence.diagonal()
        diag_rows = np.expand_dims(diag, axis=0)
        diag_cols = np.expand_dims(diag, axis=1)
        # this essentially does vstack(diag_rows).T + vstack(diag_rows) - cooccurrence
        denom = diag_rows + diag_cols - cooccurrence
        return cooccurrence / denom

    @staticmethod
    def __lift(cooccurrence):
        """Helper method to calculate the Lift of the item-item similarity"""
        diag = cooccurrence.diagonal()
        diag_rows = np.expand_dims(diag, axis=0)
        diag_cols = np.expand_dims(diag, axis=1)
        denom = diag_rows * diag_cols
        return cooccurrence / denom

    # stateful time function
    def time(self):
        """
        Time a particular section of the code - call this once to set the state somewhere
        in the code, then call it again to return the elapsed time since last call.
        Call again to set the time and so on...

        Returns:
             None if we're not in debug mode - doesn't do anything
             False if timer started
             time in seconds since the last time time function was called
        """
        if self.debug:
            from time import time

            if self.start_time is None:
                self.start_time = time()
                return False
            else:
                answer = time() - self.start_time
                # reset state
                self.start_time = None
                return answer
        else:
            return None

    def compute_affinity_matrix(self, df, n_users, n_items):
        """ Affinity matrix
        The user-affinity matrix can be constructed by treating the users and items as
        indices in a sparse matrix, and the events as the data. Here, we're treating
        the ratings as the event weights.  We convert between different sparse-matrix
        formats to de-duplicate user-item pairs, otherwise they will get added up.
        Args:
            df (pd.DataFrame): Hashed df of users and items.
            n_users (int): Number of users.
            n_items (int): Number of items.
        Returns:
            scipy.csr: Affinity matrix in Compressed Sparse Row (CSR) format.
        """
        user_affinity = (
            sparse.coo_matrix(
                (
                    df[self.col_rating],
                    (df[self._col_hashed_users], df[self._col_hashed_items]),
                ),
                shape=(n_users, n_items),
            )
            .todok()
            .tocsr()
        )
        return user_affinity

    def compute_coocurrence_matrix(self, df, n_users, n_items):
        """ Coocurrence matrix
        C = U'.transpose() * U'
        where U' is the user_affinity matrix with 1's as values (instead of ratings).
        Args:
            df (pd.DataFrame): Hashed df of users and items.
            n_users (int): Number of users.
            n_items (int): Number of items.
        Returns:
            np.array: Coocurrence matrix        
        """
        self.time()
        user_item_hits = (
            sparse.coo_matrix(
                (
                    [1] * len(df[self._col_hashed_users]),
                    (df[self._col_hashed_users], df[self._col_hashed_items]),
                ),
                shape=(n_users, n_items),
            )
            .todok()
            .tocsr()
        )

        # FIXME: workaround to avoid odd memory problem
        fname = "user_item_hits.npz"
        sparse.save_npz(fname, user_item_hits)
        user_item_hits = sparse.load_npz(fname)

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        if self.debug:
            cnt = df.shape[0]
            elapsed_time = self.time()
            self.timer_log += [
                "Item cooccurrence calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        self.time()
        item_cooccurrence = item_cooccurrence.multiply(
            item_cooccurrence >= self.threshold
        )
        if self.debug:
            elapsed_time = self.time()
            self.timer_log += [
                "Applying threshold:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]
        return item_cooccurrence

    def fit(self, df):
        """Main fit method for SAR"""

        log.info("Collecting user affinity matrix...")
        self.time()
        # use the same floating type for the computations as input
        float_type = df[self.col_rating].dtype
        if not np.issubdtype(float_type, np.floating):
            raise ValueError(
                "Only floating point data types are accepted for the rating column. Data type was {} "
                "instead.".format(float_type)
            )

        if self.timedecay_formula:
            # WARNING: previously we would take the last value in training dataframe and set it
            # as a matrix U element
            # for each user-item pair. Now with time decay, we compute a sum over ratings given
            # by a user in the case
            # when T=np.inf, so user gets a cumulative sum of ratings for a particular item and
            # not the last rating.
            log.info("Calculating time-decayed affinities...")
            # Time Decay
            # do a group by on user item pairs and apply the formula for time decay there
            # Time T parameter is in days and input time is in seconds
            # so we do dt/60/(T*24*60)=dt/(T*24*3600)

            # if time_now is None - get the default behaviour
            if not self.time_now:
                self.time_now = df[self.col_timestamp].max()

            # optimization - pre-compute time decay exponential which multiplies the ratings
            expo_fun = lambda x: np.exp(
                -np.log(2.0)
                * (self.time_now - x)
                / (self.time_decay_coefficient * 24.0 * 3600)
            )

            rating_exponential = df[self.col_rating].values * expo_fun(
                df[self.col_timestamp].values
            ).astype(float_type)
            # update df with the affinities after the timestamp calculation
            # copy part of the data frame to avoid modification of the input
            temp_df = pd.DataFrame(
                data={
                    self.col_user: df[self.col_user],
                    self.col_item: df[self.col_item],
                    self.col_rating: rating_exponential,
                }
            )
            newdf = temp_df.groupby([self.col_user, self.col_item]).sum().reset_index()

            """
            # experimental implementation of multiprocessing - in practice for smaller datasets this is not needed
            # leaving here in case anyone wants to actually try this
            # to enable, you need:
            #   conda install dill>=0.2.8.1
            #   pip install multiprocess>=0.70.6.1
            # from multiprocess import Pool, cpu_count
            # 
            # multiproces uses dill for python3 to serialize lambda functions
            #
            # helper function to parallelize the operation on groups
            def applyParallel(dfGrouped, func):
                with Pool(cpu_count()*2) as p:
                    ret_list = p.map(func, [group for name, group in dfGrouped])
                return pd.concat(ret_list)

            from types import MethodType
            grouped.applyParallel = MethodType(applyParallel, grouped)

            # then replace df.apply with df.applyParallel
            """

            """
            Original implementatoin of groupby and apply - without optimization
            rating_series = grouped.apply(lambda x: np.sum(np.array(x[self.col_rating]) * np.exp(
                -np.log(2.) * (self.time_now - np.array(x[self.col_timestamp])) / (
                    self.time_decay_coefficient * 24. * 3600))))
            """

        else:
            # without time decay we take the last user-provided rating supplied in the dataset as the
            # final rating for the user-item pair
            log.info("Deduplicating the user-item counts")
            newdf = df.drop_duplicates([self.col_user, self.col_item])[
                [self.col_user, self.col_item, self.col_rating]
            ]

        if self.debug:
            elapsed_time = self.time()
            cnt = newdf.shape[0]
            self.timer_log += [
                "Affinity calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        self.time()
        log.info("Creating index columns...")
        # Hash users and items according to the two dicts. Add the two new columns to newdf.
        newdf.loc[:, self._col_hashed_items] = newdf[self.col_item].map(
            self.item_map_dict
        )
        newdf.loc[:, self._col_hashed_users] = newdf[self.col_user].map(
            self.user_map_dict
        )

        # store training set index for future use during prediction
        # DO NOT USE .values as the warning message suggests
        self.index = newdf[[self._col_hashed_users, self._col_hashed_items]].values

        n_items = len(self.unique_items)
        n_users = len(self.unique_users)

        # Affinity matrix
        log.info("Building user affinity sparse matrix...")
        self.user_affinity = self.compute_affinity_matrix(newdf, n_users, n_items)

        if self.debug:
            elapsed_time = self.time()
            self.timer_log += [
                "Indexing and affinity matrix construction:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        # Calculate item cooccurrence
        log.info("Calculating item cooccurrence...")
        item_cooccurrence = self.compute_coocurrence_matrix(newdf, n_users, n_items)

        log.info("Calculating item similarity...")
        similarity_type = (
            SIM_COOCCUR if self.similarity_type is None else self.similarity_type
        )

        self.time()
        if similarity_type == SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif similarity_type == SIM_JACCARD:
            self.item_similarity = self.__jaccard(item_cooccurrence)
        elif similarity_type == SIM_LIFT:
            self.item_similarity = self.__lift(item_cooccurrence)
        else:
            raise ValueError("Unknown similarity type: {0}".format(similarity_type))

        self.item_similarity = self.item_similarity.astype(float_type, copy=False)

        if self.debug and (
            similarity_type == SIM_JACCARD or similarity_type == SIM_LIFT
        ):
            elapsed_time = self.time()
            self.timer_log += [
                "Item similarity calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        # Calculate raw scores with a matrix multiplication.
        log.info("Calculating recommendation scores...")
        self.time()
        self.scores = self.user_affinity.dot(self.item_similarity)

        if self.debug:
            elapsed_time = self.time()
            self.timer_log += [
                "Score calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        log.info("done training")

    def recommend_k_items(self, test, top_k=10, sort_top_k=False, **kwargs):
        """Recommend top K items for all users which are in the test set

        Args:
            **kwargs:

        Returns:
            pd.DataFrame: A DataFrame that contains top k recommendation items for each user.
        """

        # pick users from test set and
        test_users = test[self.col_user].unique()
        try:
            test_users_training_ids = np.array(
                [self.user_map_dict[user] for user in test_users]
            )
        except KeyError():
            msg = "SAR cannot score test set users which are not in the training set"
            log.error(msg)
            raise ValueError(msg)

        # shorthand
        scores = self.scores

        # Convert to dense, the following operations are easier.
        log.info("Converting to dense matrix...")
        if isinstance(scores, np.matrixlib.defmatrix.matrix):
            scores_dense = np.array(scores)
        else:
            scores_dense = scores.todense()

        # Mask out items in the train set.  This only makes sense for some
        # problems (where a user wouldn't interact with an item more than once).
        if self.remove_seen:
            log.info("Removing seen items...")
            scores_dense[self.index[:, 0], self.index[:, 1]] = 0

        # Get top K items and scores.
        log.info("Getting top K...")
        top_items = np.argpartition(scores_dense, -top_k, axis=1)[:, -top_k:]
        top_scores = scores_dense[np.arange(scores_dense.shape[0])[:, None], top_items]

        log.info("Select users from the test set")
        top_items = top_items[test_users_training_ids, :]
        top_scores = top_scores[test_users_training_ids, :]

        log.info("Creating output dataframe...")

        # Convert to np.array (from view) and flatten
        top_items = np.reshape(np.array(top_items), -1)
        top_scores = np.reshape(np.array(top_scores), -1)

        userids = []
        for u in test_users:
            userids.extend([u] * top_k)

        results = pd.DataFrame.from_dict(
            {
                self.col_user: userids,
                self.col_item: top_items,
                self.col_rating: top_scores,
            }
        )

        # remap user and item indices to IDs
        results[self.col_item] = results[self.col_item].map(self.index2item)

        # do final sort
        if sort_top_k:
            results = (
                results.sort_values(
                    by=[self.col_user, self.col_rating], ascending=False
                )
                .groupby(self.col_user)
                .apply(lambda x: x)
            )

        # format the dataframe in the end to conform to Suprise return type
        log.info("Formatting output")

        # modify test to make it compatible with

        return (
            results[[self.col_user, self.col_item, self.col_rating]]
            .rename(columns={self.col_rating: PREDICTION_COL})
            .astype(
                {
                    self.col_user: _user_item_return_type(),
                    self.col_item: _user_item_return_type(),
                    PREDICTION_COL: self.scores.dtype,
                }
            )
        )

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set

        Args:
            test (pd.DataFrame): DataFrame that contains ground-truth of user-item ratings.

        Return:
            pd.DataFrame: DataFrame contains the prediction results.
        """
        # pick users from test set and
        test_users = test[self.col_user].unique()
        try:
            training_ids = np.array([self.user_map_dict[user] for user in test_users])
            assert training_ids is not None
        except KeyError():
            msg = "SAR cannot score test set users which are not in the training set"
            log.error(msg)
            raise ValueError(msg)

        # shorthand
        scores = self.scores

        # Convert to dense, the following operations are easier.
        log.info("Converting to dense matrix...")
        if isinstance(scores, np.matrixlib.defmatrix.matrix):
            scores_dense = np.array(scores)
        else:
            scores_dense = scores.todense()

        # take the intersection between train test items and items we actually need
        test_col_hashed_users = test[self.col_user].map(self.user_map_dict)
        test_col_hashed_items = test[self.col_item].map(self.item_map_dict)

        test_index = pd.concat(
            [test_col_hashed_users, test_col_hashed_items], axis=1
        ).values
        aset = set([tuple(x) for x in self.index])
        bset = set([tuple(x) for x in test_index])

        common_index = np.array([x for x in aset & bset])

        # Mask out items in the train set.  This only makes sense for some
        # problems (where a user wouldn't interact with an item more than once).
        if self.remove_seen and len(aset & bset) > 0:
            log.info("Removing seen items...")
            scores_dense[common_index[:, 0], common_index[:, 1]] = 0

        final_scores = scores_dense[test_index[:, 0], test_index[:, 1]]

        results = pd.DataFrame.from_dict(
            {
                self.col_user: test_index[:, 0],
                self.col_item: test_index[:, 1],
                self.col_rating: final_scores,
            }
        )

        # remap user and item indices to IDs
        results[self.col_user] = results[self.col_user].map(self.index2user)
        results[self.col_item] = results[self.col_item].map(self.index2item)

        # format the dataframe in the end to conform to Suprise return type
        log.info("Formatting output")

        # modify test to make it compatible with
        return (
            results[[self.col_user, self.col_item, self.col_rating]]
            .rename(columns={self.col_rating: PREDICTION_COL})
            .astype(
                {
                    self.col_user: _user_item_return_type(),
                    self.col_item: _user_item_return_type(),
                    PREDICTION_COL: self.scores.dtype,
                }
            )
        )


def _user_item_return_type():
    return str
