# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
import logging
from scipy import sparse

from recommenders.utils.python_utils import (
    jaccard,
    lift,
    exponential_decay,
    get_top_k_scored_items,
    rescale,
)
from recommenders.utils import constants


COOCCUR = "cooccurrence"
JACCARD = "jaccard"
LIFT = "lift"

logger = logging.getLogger()


class SARSingleNode:
    """Simple Algorithm for Recommendations (SAR) implementation

    SAR is a fast scalable adaptive algorithm for personalized recommendations based on user transaction history
    and items description. The core idea behind SAR is to recommend items like those that a user already has
    demonstrated an affinity to. It does this by 1) estimating the affinity of users for items, 2) estimating
    similarity across items, and then 3) combining the estimates to generate a set of recommendations for a given user.
    """

    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_rating=constants.DEFAULT_RATING_COL,
        col_timestamp=constants.DEFAULT_TIMESTAMP_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        similarity_type=JACCARD,
        time_decay_coefficient=30,
        time_now=None,
        timedecay_formula=False,
        threshold=1,
        normalize=False,
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            col_prediction (str): prediction column name
            similarity_type (str): ['cooccurrence', 'jaccard', 'lift'] option for computing item-item similarity
            time_decay_coefficient (float): number of days till ratings are decayed by 1/2
            time_now (int | None): current time for time decay calculation
            timedecay_formula (bool): flag to apply time decay
            threshold (int): item-item co-occurrences below this threshold will be removed
            normalize (bool): option for normalizing predictions to scale of original ratings
        """
        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        self.col_timestamp = col_timestamp
        self.col_prediction = col_prediction

        if similarity_type not in [COOCCUR, JACCARD, LIFT]:
            raise ValueError(
                'Similarity type must be one of ["cooccurrence" | "jaccard" | "lift"]'
            )
        self.similarity_type = similarity_type
        self.time_decay_half_life = (
            time_decay_coefficient * 24 * 60 * 60
        )  # convert to seconds
        self.time_decay_flag = timedecay_formula
        self.time_now = time_now
        self.threshold = threshold
        self.user_affinity = None
        self.item_similarity = None
        self.item_frequencies = None

        # threshold - items below this number get set to zero in co-occurrence counts
        if self.threshold <= 0:
            raise ValueError("Threshold cannot be < 1")

        # set flag to capture unity-rating user-affinity matrix for scaling scores
        self.normalize = normalize
        self.col_unity_rating = "_unity_rating"

        # column for mapping user / item ids to internal indices
        self.col_item_id = "_indexed_items"
        self.col_user_id = "_indexed_users"

        # obtain all the users and items from both training and test data
        self.n_users = None
        self.n_items = None

        # The min and max of the rating scale, obtained from the training data.
        self.rating_min = None
        self.rating_max = None

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above map - map array index to actual string ID
        self.index2item = None

    def compute_affinity_matrix(self, df, rating_col):
        """Affinity matrix.

        The user-affinity matrix can be constructed by treating the users and items as
        indices in a sparse matrix, and the events as the data. Here, we're treating
        the ratings as the event weights.  We convert between different sparse-matrix
        formats to de-duplicate user-item pairs, otherwise they will get added up.

        Args:
            df (pandas.DataFrame): Indexed df of users and items
            rating_col (str): Name of column to use for ratings

        Returns:
            sparse.csr: Affinity matrix in Compressed Sparse Row (CSR) format.
        """

        return sparse.coo_matrix(
            (df[rating_col], (df[self.col_user_id], df[self.col_item_id])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

    def compute_time_decay(self, df, decay_column):
        """Compute time decay on provided column.

        Args:
            df (pandas.DataFrame): DataFrame of users and items
            decay_column (str): column to decay

        Returns:
            pandas.DataFrame: with column decayed
        """

        # if time_now is None use the latest time
        if self.time_now is None:
            self.time_now = df[self.col_timestamp].max()

        # apply time decay to each rating
        df[decay_column] *= exponential_decay(
            value=df[self.col_timestamp],
            max_val=self.time_now,
            half_life=self.time_decay_half_life,
        )

        # group time decayed ratings by user-item and take the sum as the user-item affinity
        return df.groupby([self.col_user, self.col_item]).sum().reset_index()

    def compute_coocurrence_matrix(self, df):
        """Co-occurrence matrix.

        The co-occurrence matrix is defined as :math:`C = U^T * U`

        where U is the user_affinity matrix with 1's as values (instead of ratings).

        Args:
            df (pandas.DataFrame): DataFrame of users and items

        Returns:
            numpy.ndarray: Co-occurrence matrix
        """

        user_item_hits = sparse.coo_matrix(
            (np.repeat(1, df.shape[0]), (df[self.col_user_id], df[self.col_item_id])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        item_cooccurrence = item_cooccurrence.multiply(
            item_cooccurrence >= self.threshold
        )

        return item_cooccurrence.astype(df[self.col_rating].dtype)

    def set_index(self, df):
        """Generate continuous indices for users and items to reduce memory usage.

        Args:
            df (pandas.DataFrame): dataframe with user and item ids
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
        """Main fit method for SAR.
        
        .. note::

        Please make sure that `df` has no duplicates.

        Args:
            df (pandas.DataFrame): User item rating dataframe (without duplicates).
        """

        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        logger.info("Collecting user affinity matrix")
        if not np.issubdtype(df[self.col_rating].dtype, np.number):
            raise TypeError("Rating column data type must be numeric")

        # copy the DataFrame to avoid modification of the input
        select_columns = [self.col_user, self.col_item, self.col_rating]
        if self.time_decay_flag:
            select_columns += [self.col_timestamp]
        temp_df = df[select_columns].copy()

        if self.time_decay_flag:
            logger.info("Calculating time-decayed affinities")
            temp_df = self.compute_time_decay(df=temp_df, decay_column=self.col_rating)

        logger.info("Creating index columns")
        # add mapping of user and item ids to indices
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].apply(
            lambda item: self.item2index.get(item, np.NaN)
        )
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].apply(
            lambda user: self.user2index.get(user, np.NaN)
        )

        if self.normalize:
            self.rating_min = temp_df[self.col_rating].min()
            self.rating_max = temp_df[self.col_rating].max()
            logger.info("Calculating normalization factors")
            temp_df[self.col_unity_rating] = 1.0
            if self.time_decay_flag:
                temp_df = self.compute_time_decay(
                    df=temp_df, decay_column=self.col_unity_rating
                )
            self.unity_user_affinity = self.compute_affinity_matrix(
                df=temp_df, rating_col=self.col_unity_rating
            )

        # affinity matrix
        logger.info("Building user affinity sparse matrix")
        self.user_affinity = self.compute_affinity_matrix(
            df=temp_df, rating_col=self.col_rating
        )

        # calculate item co-occurrence
        logger.info("Calculating item co-occurrence")
        item_cooccurrence = self.compute_coocurrence_matrix(df=temp_df)

        # free up some space
        del temp_df

        self.item_frequencies = item_cooccurrence.diagonal()

        logger.info("Calculating item similarity")
        if self.similarity_type is COOCCUR:
            logger.info("Using co-occurrence based similarity")
            self.item_similarity = item_cooccurrence
        elif self.similarity_type is JACCARD:
            logger.info("Using jaccard based similarity")
            self.item_similarity = jaccard(item_cooccurrence).astype(
                df[self.col_rating].dtype
            )
        elif self.similarity_type is LIFT:
            logger.info("Using lift based similarity")
            self.item_similarity = lift(item_cooccurrence).astype(
                df[self.col_rating].dtype
            )
        else:
            raise ValueError("Unknown similarity type: {}".format(self.similarity_type))

        # free up some space
        del item_cooccurrence

        logger.info("Done training")

    def score(self, test, remove_seen=False):
        """Score all items for test users.

        Args:
            test (pandas.DataFrame): user to test
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            numpy.ndarray: Value of interest of all items for the users.
        """

        # get user / item indices from test set
        user_ids = list(
            map(
                lambda user: self.user2index.get(user, np.NaN),
                test[self.col_user].unique(),
            )
        )
        if any(np.isnan(user_ids)):
            raise ValueError("SAR cannot score users that are not in the training set")

        # calculate raw scores with a matrix multiplication
        logger.info("Calculating recommendation scores")
        test_scores = self.user_affinity[user_ids, :].dot(self.item_similarity)

        # ensure we're working with a dense ndarray
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.toarray()

        if self.normalize:
            counts = self.unity_user_affinity[user_ids, :].dot(self.item_similarity)
            user_min_scores = (
                np.tile(counts.min(axis=1)[:, np.newaxis], test_scores.shape[1])
                * self.rating_min
            )
            user_max_scores = (
                np.tile(counts.max(axis=1)[:, np.newaxis], test_scores.shape[1])
                * self.rating_max
            )
            test_scores = rescale(
                test_scores,
                self.rating_min,
                self.rating_max,
                user_min_scores,
                user_max_scores,
            )

        # remove items in the train set so recommended items are always novel
        if remove_seen:
            logger.info("Removing seen items")
            test_scores += self.user_affinity[user_ids, :] * -np.inf

        return test_scores

    def get_popularity_based_topk(self, top_k=10, sort_top_k=True):
        """Get top K most frequently occurring items across all users.

        Args:
            top_k (int): number of top items to recommend.
            sort_top_k (bool): flag to sort top k results.

        Returns:
            pandas.DataFrame: top k most popular items.
        """

        test_scores = np.array([self.item_frequencies])

        logger.info("Getting top K")
        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        return pd.DataFrame(
            {
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

    def get_item_based_topk(self, items, top_k=10, sort_top_k=True):
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
            items (pandas.DataFrame): DataFrame with item, user (optional), and rating (optional) columns
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results

        Returns:
            pandas.DataFrame: sorted top k recommendation items
        """

        # convert item ids to indices
        item_ids = np.asarray(
            list(
                map(
                    lambda item: self.item2index.get(item, np.NaN),
                    items[self.col_item].values,
                )
            )
        )

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

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test_users.drop_duplicates().values, top_items.shape[1]
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def recommend_k_items(self, test, top_k=10, sort_top_k=True, remove_seen=False):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pandas.DataFrame): users to test
            top_k (int): number of top items to recommend
            sort_top_k (bool): flag to sort top k results
            remove_seen (bool): flag to remove items seen in training from recommendation

        Returns:
            pandas.DataFrame: top k recommendation items for each user
        """

        test_scores = self.score(test, remove_seen=remove_seen)

        top_items, top_scores = get_top_k_scored_items(
            scores=test_scores, top_k=top_k, sort_top_k=sort_top_k
        )

        df = pd.DataFrame(
            {
                self.col_user: np.repeat(
                    test[self.col_user].drop_duplicates().values, top_items.shape[1]
                ),
                self.col_item: [self.index2item[item] for item in top_items.flatten()],
                self.col_prediction: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set

        Args:
            test (pandas.DataFrame): DataFrame that contains users and items to test

        Returns:
            pandas.DataFrame: DataFrame contains the prediction results
        """

        test_scores = self.score(test)
        user_ids = np.asarray(
            list(
                map(
                    lambda user: self.user2index.get(user, np.NaN),
                    test[self.col_user].values,
                )
            )
        )

        # create mapping of new items to zeros
        item_ids = np.asarray(
            list(
                map(
                    lambda item: self.item2index.get(item, np.NaN),
                    test[self.col_item].values,
                )
            )
        )
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
