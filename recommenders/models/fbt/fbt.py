# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from recommenders.utils import constants
from recommenders.utils.python_utils import get_top_k_scored_items
import pandas as pd
import numpy as np
import logging
from scipy import sparse

logger = logging.getLogger()


class FBT(object):
    """Frequently Bought Together Algorithm for Recommendations
    (FBT) implementation

    FBT is a fast scalable adaptive algorithm for personalized
    recommendations based on user-item interaction history.
    The core idea behind FBT: Given that a user has interacted
    with an item, what other items were most frequently interacted with
    by other users? Lets recommend these other items to our current user!
    """

    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
        """
        self.col_item = col_item
        self.col_user = col_user

        # column for mapping user / item ids to internal indices
        self.col_item_id = "_indexed_items"
        self.col_user_id = "_indexed_users"

        # obtain all the users and items from both training and test data
        self.n_users = None
        self.n_items = None

        # mapping for item to matrix element
        self.user2index = None
        self.item2index = None

        # the opposite of the above map - map array index to actual string ID
        self.index2item = None

        self.user_affinity = None
        self._item_similarity = None
        self._item_frequencies = None

        # set flag to disallow calling fit() before predict()
        self._is_fit = False

    def __repr__(self):
        """Make a friendly, human-readable object summary string."""
        return (
            f"{self.__class__.__name__}(user_colname={self.col_user}, "
            f"item_colname={self.col_item}, "
            f"score_colname={self.col_score} "
        )

    def _check_dataframe(self, df,
                         expected_columns=None):
        """Verify input is a dataframe and has expected columns
           and if there are duplicate rows.

        Args:
            df (pandas.DataFrame): Input dataframe

            expected_columns (list()): List of expected column names
                                       for the dataframe

        Returns:
            TypeError, ValueError or KeyError if checks fail.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError((
                "Input argument must be a pandas DataFrame. "
                f"Instead got {type(df)}!"))

        if not expected_columns:
            expected_columns = [self.col_user, self.col_item]
        for col in expected_columns:
            if col not in df.columns:
                raise KeyError((
                    f"Column {col} not found in DataFrame!"))

        # check there are no duplicate rows in dataframe
        if np.any(df.duplicated()):
            raise ValueError("Duplicate rows found!!")

    def compute_affinity_matrix(self, df):
        """Affinity matrix.

        The user-affinity matrix can be constructed by treating the users and items as
        indices in a sparse matrix, and the events as the data. Here, we treat
        a user-item interactions as the event weights, so 1 if user interacted with
        the item, 0 otherwise.  We convert between different sparse-matrix
        formats to de-duplicate user-item pairs, otherwise they will get added up.

        Args:
            df (pandas.DataFrame): Indexed df of users and items

        Returns:
            sparse.csr: Affinity matrix in Compressed Sparse Row (CSR) format.
        """
        user_item_hits = sparse.coo_matrix(
            (np.repeat(1, df.shape[0]), (df[self.col_user_id], df[self.col_item_id])),
            shape=(self.n_users, self.n_items),
        ).tocsr()

        return user_item_hits

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

    def compute_coocurrence_matrix(self, df):
        """ Co-occurrence matrix.
        The co-occurrence matrix is defined as :math:`C = U^T * U`
        where U is the user_affinity matrix with 1's as values (instead of ratings).
        Args:
            df (pandas.DataFrame): DataFrame of users and items
        Returns:
            numpy.ndarray: Co-occurrence matrix
        """

        user_item_hits = self.compute_affinity_matrix(df)

        item_cooccurrence = user_item_hits.transpose().dot(user_item_hits)
        return item_cooccurrence.astype(np.number)

    def fit(self, df):
        """Fit the FBT recommender using an input DataFrame.

        Fitting of model involves computing a item-item co-occurrence
        matrix: how many people watched a pair of movies?

        Args:
            df (pd.DataFrame): DataFrame of users and items

        """
        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        # Only choose the user and item columns from the input
        select_columns = [self.col_user, self.col_item]
        temp_df = df[select_columns].copy()

        logger.info("Check input dataframe to fit() is of the type, schema "
                    "we expect and there are no duplicates.")
        self._check_dataframe(temp_df)

        logger.info("Creating index columns")
        # add mapping of user and item ids to indices
        temp_df.loc[:, self.col_item_id] = temp_df[self.col_item].apply(
            lambda item: self.item2index.get(item, np.NaN)
        )
        temp_df.loc[:, self.col_user_id] = temp_df[self.col_user].apply(
            lambda user: self.user2index.get(user, np.NaN)
        )

        # affinity matrix
        logger.info("Building user affinity sparse matrix")
        self.user_affinity = self.compute_affinity_matrix(df=temp_df)

        # similarity score is defined by how many distinct
        # users interacted with the same pair of items (cooccurrence)
        self.item_similarity = self.compute_coocurrence_matrix(df=temp_df)

        # free up some space
        del temp_df

        # Item frequencies can be obtained by looking at the
        # number of distinct users who interacted with a specific
        # item: can be extracted from the diagonal of the similarity matrix
        self.item_frequencies = self.item_similarity.diagonal()

        self._is_fit = True
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
            raise ValueError("FBT cannot score users that are not in the training set")

        # calculate raw scores with a matrix multiplication
        logger.info("Calculating recommendation scores")
        test_scores = self.user_affinity[user_ids, :].dot(self.item_similarity)

        # ensure we're working with a dense ndarray
        if isinstance(test_scores, sparse.spmatrix):
            test_scores = test_scores.toarray()

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
                self.col_score: top_scores.flatten(),
            }
        )

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
                self.col_score: top_scores.flatten(),
            }
        )

        # drop invalid items
        return df.replace(-np.inf, np.nan).dropna()

    def predict(self, test):
        """Output FBT scores for only the users-items pairs which are in the test set

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
                self.col_score: test_scores[user_ids, item_ids],
            }
        )
        return df

    # def predict(self, test):
    #     """Generate new recommendations using a trained FBT model.

    #     Args
    #     ----
    #     test : pandas.DataFrame
    #         DataFrame of users and items with each row being a unique pair.

    #     Returns
    #     -------
    #     pandas.DataFrame
    #         DataFrame with each row is a recommendations for each user in X.
    #     """
    #     if not self._is_fit:
    #         raise ValueError("fit() must be called before predict()!")

    #     logger.info("Check input dataframe to predict() is of type, schema "
    #                 "we expect and there are no duplicates.")

    #     self._check_dataframe(test)
    #     logger.info("Calculating recommendation scores")

    #     # To get recommendations, merge test dataset of users with the
    #     # item similarity on col_item to get all matches
    #     all_test_item_matches_df = test.merge(
    #         self._model_df,
    #         on=self.col_item,
    #         how='left'
    #     )
    #     # Give user may interact with multiple items A, B, C, a
    #     # item D may be recommended as a popular paired choice for
    #     # each of A, B, C. Will average such scores
    #     all_recommendations_df = (
    #         all_test_item_matches_df
    #         .groupby([self.col_user,
    #                  f'{self.col_item}_paired'])[self.col_score]
    #         .mean()
    #         .reset_index(drop=False)
    #         .rename(columns={f'{self.col_item}_paired': self.col_item})
    #     )
    #     all_recommendations_df[self.col_item] = (
    #         all_recommendations_df[self.col_item].astype('int64')
    #     )

    #     return all_recommendations_df

    # def recommend_k_items(self,
    #                       test,
    #                       remove_seen=False,
    #                       train=None,
    #                       top_k=10):
    #     """Recommend top K items for all users which are in the test set

    #     Args:
    #         test (pd.DataFrame): users to test
    #         top_k (int): number of top items to recommend
    #         remove_seen (bool): flag to remove recommendations
    #         that have been seen by user

    #     Returns:
    #         pandas.DataFrame: top k recommended items for each user
    #     """

    #     logger.info(f"Recommending top {top_k} items for each user...")
    #     all_recommendations_df = self.predict(test)

    #     if remove_seen:
    #         if train is None:
    #             raise ValueError("Please provide the training data to "
    #                              "remove seen items!")
    #         # select user-item columns of the DataFrame
    #         select_columns = [self.col_user, self.col_item]
    #         temp_df = train[select_columns]

    #         logger.info("Check train dataframe is of the type, schema "
    #                     "we expect and there are no duplicates.")

    #         self._check_dataframe(train)

    #         seen_item_indicator = (
    #             all_recommendations_df
    #             .merge(temp_df,
    #                    on=select_columns,
    #                    how='left',
    #                    indicator=True)
    #         )
    #         # filtering original recommendations to novel ones
    #         recommendations_df = (
    #             seen_item_indicator
    #             .loc[seen_item_indicator['_merge'] == 'left_only']
    #         )

    #         recommendations_df.drop(columns=['_merge'], inplace=True)
    #     else:
    #         recommendations_df = all_recommendations_df

    #     topk_recommendations_df = get_top_k_scored_items(recommendations_df,
    #                                                      col_user=self.col_user,
    #                                                      col_rating=self.col_score,
    #                                                      k=top_k)

    #     # Making sure we have a row for every test user even if null
    #     test_users = pd.DataFrame(set(test[self.col_user]),
    #                               columns=[self.col_user])
    #     final_k_recommendations = (
    #         test_users
    #         .merge(topk_recommendations_df,
    #                on=self.col_user,
    #                how='left')
    #     )

    #     return final_k_recommendations
