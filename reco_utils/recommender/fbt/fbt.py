# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from reco_utils.common import constants
from reco_utils.evaluation.python_evaluation import get_top_k_items
import pandas as pd
import logging

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
        col_item=constants.DEFAULT_ITEM_COL,
        col_score=constants.DEFAULT_PREDICTION_COL
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_score (str): score column name
        """
        self.col_item = col_item
        self.col_user = col_user
        self.col_score = col_score

        self._model_df = None
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

        Args
        ----
        df (pandas.DataFrame): Input dataframe

        expected_columns: list()
            List of expected column names for the dataframe

        Returns
        -------
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
        if df.groupby(expected_columns).size().max() > 1:
            raise ValueError("Duplicate rows found!!")

    def fit(self, df):
        """Fit the FBT recommender using an input DataFrame.

        Fitting of model involves computing a item-item co-occurrence
        matrix: how many people bought watched a pair of movies? This
        matrix is an attribute of the class object.

        Args:
            df (pd.DataFrame): DataFrame of users and items

        """
        # Only choose the user and item columns from the input
        select_columns = [self.col_user, self.col_item]
        temp_df = df[select_columns]

        logger.info("Check input dataframe to fit() is of the type, schema "
                    "we expect and there are no duplicates.")
        self._check_dataframe(df)

        # To compute co-occurrence, key piece is the self-join
        cooccurrence_df = (
            temp_df
            .merge(temp_df, on=self.col_user,
                   how='inner', suffixes=['', '_paired'])
        )
        # similarity score is defined by how many distinct
        # users interacted with the same pair of items
        sim_df = (
            cooccurrence_df
            .groupby([self.col_item,
                     f'{self.col_item}_paired'])[self.col_user]
            .nunique()
            .reset_index(drop=False)
            .rename(columns={self.col_user: self.col_score})
        )

        # Retaining item-pairs where item isn't paired with itself
        fbt_df = (
            sim_df
            .loc[sim_df
                 [self.col_item] !=
                 sim_df[f'{self.col_item}_paired']]
        )

        self._model_df = fbt_df

        # Item frequencies can be obtained by looking at the
        # number of distinct users who interacted with a specific
        # item: can be extracted from items paired with itself
        self.item_frequencies = (
            sim_df
            .loc[sim_df
                 [self.col_item] ==
                 sim_df[f'{self.col_item}_paired']]
            .drop(columns=[f'{self.col_item}_paired'])
        )

        self._is_fit = True
        logger.info("Done training")

    def predict(self, test):
        """Generate new recommendations using a trained FBT model.

        Args
        ----
        test : pandas.DataFrame
            DataFrame of users and items with each row being a unique pair.

        Returns
        -------
        pandas.DataFrame
            DataFrame with each row is a recommendations for each user in X.
        """
        if not self._is_fit:
            raise ValueError("fit() must be called before predict()!")

        logger.info("Check input dataframe to predict() is of the type, schema "
                    "we expect and there are no duplicates.")

        self._check_dataframe(test)
        logger.info("Calculating recommendation scores")

        # To get recommendations, merge test dataset of users with the
        # item similarity on col_item to get all matches
        all_test_item_matches_df = test.merge(
            self._model_df,
            on=self.col_item,
            how='left'
        )
        # Give user may interact with multiple items A, B, C, a
        # item D may be recommended as a popular paired choice for
        # each of A, B, C. Will average such scores
        all_recommendations_df = (
            all_test_item_matches_df
            .groupby([self.col_user,
                     f'{self.col_item}_paired'])[self.col_score]
            .mean()
            .reset_index(drop=False)
            .rename(columns={f'{self.col_item}_paired': self.col_item})
        )
        all_recommendations_df[self.col_item] = (
            all_recommendations_df[self.col_item].astype('int64')
        )

        return all_recommendations_df

    def recommend_k_items(self,
                          test,
                          remove_seen=False,
                          train=None,
                          top_k=10):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pd.DataFrame): users to test
            top_k (int): number of top items to recommend
            remove_seen (bool): flag to remove recommendations
            that have been seen by user

        Returns:
            pandas.DataFrame: top k recommended items for each user
        """

        logger.info(f"Recommending top {top_k} items for each user...")
        all_recommendations_df = self.predict(test)

        if remove_seen:
            if train is None:
                raise ValueError("Please provide the training data to "
                                 "remove seen items!")
            # select user-item columns of the DataFrame
            select_columns = [self.col_user, self.col_item]
            temp_df = train[select_columns]

            seen_item_indicator = (
                all_recommendations_df
                .merge(temp_df,
                       on=select_columns,
                       how='left',
                       indicator=True)
            )
            # filtering original recommendations to novel ones
            recommendations_df = (
                seen_item_indicator
                .loc[seen_item_indicator['_merge'] == 'left_only']
            )

            recommendations_df.drop(columns=['_merge'], inplace=True)
        else:
            recommendations_df = all_recommendations_df

        topk_recommendations_df = get_top_k_items(recommendations_df,
                                                  col_user=self.col_user,
                                                  col_rating=self.col_score,
                                                  k=top_k)

        # Making sure we have a row for every test user even if null
        test_users = test[self.col_user].unique()
        final_k_recommendations = (
            test_users
            .merge(topk_recommendations_df,
                   on=self.col_user,
                   how='left')
        )

        return final_k_recommendations
