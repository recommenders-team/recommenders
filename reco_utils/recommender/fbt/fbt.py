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
        col_score=constants.DEFAULT_PREDICTION_COL,
        num_recos=10
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
        self.num_recos = num_recos

        self._model_df = None
        self._item_frequencies = None

        # set flag to disallow calling fit() before predict()
        self._is_fit = False

    def __repr__(self):
        """Make a friendly, human-readable object summary string."""
        return (
            f"{self.__class__.__name__}(user_colname={self.col_user}, "
            f"item_colname={self.col_item}, "
            f"score_colname={self.col_score}, "
            f"num_recos={self.num_recos}, "
        )

    def _check_dataframe(self, df,
                         expected_columns=None):
        """Verify input is a data frame and has expected columns.

        Checks if the input is a DataFrame object and if so,
        if the mandatory columns are present. Also checks for
        duplicate rows.
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
        matrix: how many people bought watched a pair of movies?

        Args:
            df (pd.DataFrame): DataFrame of users and items

        Returns:
        self
            Fitted estimator.
        """
        logger.info("Check dataframe is of the type, schema we expect")
        expected_cols_df = [self.col_user,
                            self.col_item,
                            f'{self.col_item}_name']
        self._check_dataframe(df, expected_columns=expected_cols_df)

        # copy the DataFrame to avoid modification of the input
        select_columns = [self.col_user, self.col_item]
        temp_df = df[select_columns].copy()

        logger.info("De-duplicating the user-item counts")
        temp_df = temp_df.drop_duplicates(
                [self.col_user, self.col_item], keep="last"
        )

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

        Parameters
        ----------
        test : DataFrame
            DataFrame of users who need to get recommendations
        remove_seen (bool): flag to remove items seen in training
        from recommendation

        Returns
        -------
        R : DataFrame
            DataFrame with up to `num_recos` rows per user in X.
        """
        if not self._is_fit:
            raise ValueError(("fit() must be called before predict()!"))

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
                          top_k=None):
        """Recommend top K items for all users which are in the test set

        Args:
            test (pd.DataFrame): users to test
            top_k (int): number of top items to recommend
            remove_seen (bool): flag to remove recommendations
            that have been seen by user

        Returns:
            pd.DataFrame: top k recommendation items for each user
        """

        test_users = test[[self.col_user]].drop_duplicates()
        all_recommendations_df = self.predict(test)

        if remove_seen:
            if train is None:
                raise ValueError("Please provide the training data to "
                                 "remove seen items!")
            # copy the DataFrame to avoid modification of the input
            select_columns = [self.col_user, self.col_item]
            temp_df = train[select_columns].copy()

            logger.info("De-duplicating the user-item counts")
            temp_df = temp_df.drop_duplicates(
                select_columns, keep="last"
            )

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
            # remove dataframe to save memory
            del temp_df
        else:
            recommendations_df = all_recommendations_df

        # now only keep top_k
        if not top_k:
            top_k = self.num_recos

        topk_recommendations_df = get_top_k_items(recommendations_df,
                                                  col_user=self.col_user,
                                                  col_rating=self.col_score,
                                                  k=top_k)

        # Making sure we have a row for every test user even if null
        final_k_recommendations = (
            test_users
            .merge(topk_recommendations_df,
                   on=self.col_user,
                   how='left')
        )

        return final_k_recommendations

    def eval_map_at_k(self, df_true, df_pred):  # noqa: N803
        """Evaluate quality of recommendations.

        Parameters
        ----------
        df_true: DataFrame
           DataFrame of users and items that these users have interacted with.

        df_pred: DataFrame
            DataFrame of the same users with their recommendations

        Returns
        -------
        map_at_k : float
            Computes the mean average precision at K (MAP@K) metric over all
            users in df_true. To compute the metric: if at least one of items
            bought by a user (in df_true) is found in the Top_K predicted
            recommendations served to the same user (in df_pred), we consider
            the prediction a success (1) else a failure (0).
        """
        self._check_dataframe(df_true)
        expected_columns_x_pred = [self.col_user,
                                   self.col_item,
                                   self.col_score]
        self._check_dataframe(df_pred, expected_columns_x_pred)

        preds_df = df_true.merge(
            df_pred,
            on=[self.col_user, self.col_item],
            how='left'
        )

        preds_df['is_match'] = preds_df['score'].notna().astype(int)

        # aggregate to user-level
        user_level_preds_df = (
            preds_df
            .groupby(self.col_user)
            .agg(was_match_found=('is_match', 'max'))
        )

        map_at_k = user_level_preds_df.mean()[0]

        return map_at_k
