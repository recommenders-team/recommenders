# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import logging

from reco_utils.common import constants


COOCCUR = "cooccurrence"
LIFT = "lift"
JACCARD = "jaccard"

logger = logging.getLogger()


class FBT(object):
    """Frequently Bought Together Algorithm for Recommendations (FBT) implementation

    FBT is a fast scalable adaptive algorithm for personalized recommendations based
    on user-item interaction history. The core idea behind FBT: Given that a user has
    interacted with an item, what other items were most frequently bought together
    by other users? Lets recommend these other items to our current user!
    """

    def __init__(
        self,
        col_user=constants.DEFAULT_USER_COL,
        col_item=constants.DEFAULT_ITEM_COL,
        col_prediction=constants.DEFAULT_PREDICTION_COL,
        num_recos = 10
    ):
        """Initialize model parameters

        Args:
            col_user (str): user column name
            col_item (str): item column name
            col_prediction (str): prediction column name
            num_recos (int): number of recommendations to return
        """
        self.col_item = col_item
        self.col_user = col_user
        self.col_prediction = col_prediction
        self.num_recos = num_recos

        self._model_df = None
        self._item_frequencies = None

        # set flag to disallow calling fit() before predict()
        self._is_fit = False

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

    def __repr__(self):
        """Make a friendly, human-readable object summary string."""
        return (
            f"{self.__class__.__name__}(user_colname={self.col_user}, "
            f"item_colname={self.col_item}, "
            f"prediction_colname={self.col_prediction}, "
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

    def set_index(self, df):
        """Generate continuous indices for users and items to reduce memory usage.
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
        """Fit the FBT recommender using an input DataFrame.

        Fitting of model involves computing a item-item co-occurrence
        matrix: how many people bought watched a pair of movies?

        Args:
            df (pd.DataFrame): DataFrame of users and items

        Returns:
        self
            Fitted estimator.
        """

        # generate continuous indices if this hasn't been done
        if self.index2item is None:
            self.set_index(df)

        # copy the DataFrame to avoid modification of the input
        select_columns = [self.col_user, self.col_item]
        temp_df = df[select_columns].copy()

        logger.info("De-duplicating the user-item counts")
        temp_df = temp_df.drop_duplicates(
                [self.col_user, self.col_item], keep="last"
        )

        logger.info("Check dataframe is of the type, schema we expect")
        self._check_dataframe(temp_df)

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
            .rename(columns={self.col_user: 'score'})
        )
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

        # exclude rows where course is paired with itself
        fbt_df = (
            sim_df
            .loc[sim_df
                 [self.col_item] !=
                 sim_df[f'{self.col_item}_paired']]
        )
        # for each course, add rank of all other courses
        fbt_df['rank'] = (
            fbt_df
            .groupby(self.col_item)['score']
            .rank('dense', ascending=False)
            .astype('int64')
        )

        self._model_df = fbt_df
        self._is_fit = True

        logger.info("Done training")

    def predict(self, test, remove_seen=False):
        """Generate new recommendations using a trained FBT model.

        Parameters
        ----------
        test : DataFrame
            DataFrame of users who need to get recommendations
        remove_seen (bool): flag to remove items seen in training from recommendation

        Returns
        -------
        R : DataFrame
            DataFrame with up to `num_recos_to_return` rows per user in X.
        """
        if not self._is_fit:
            raise ValueError(("fit() must be called before score()!"))

        self._check_dataframe(test)
        # get user / item indices from test set
        user_ids = list(
            map(
                lambda user: self.user2index.get(user, np.NaN),
                test[self.col_user].unique(),
            )
        )
        if any(np.isnan(user_ids)):
            raise ValueError("FBT cannot score users that are not in the training set")

        logger.info("Calculating recommendation scores")

        # start with limiting the fbt_model table to top k matches
        fbt_topk = (
            self._model_df
            .loc[self._model_df['rank'] <= self.num_recos]
            .sort_values([self.col_item, 'rank'])
        )
        # To get recommendations, merge first test dataset of users with the
        # model on col_item to get all matches
        all_preds_df = test.merge(
            fbt_topk,
            on=self.col_item,
            how='left'
        )
        # same course may be recommended for multiple courses
        # that learner already viewed, will average such scores
        topk_preds_df = (
            all_preds_df
            .groupby([self.col_user,
                     f'{self.col_item}_paired'])['score']
            .mean()
            .reset_index(drop=False)
        )
        # now only keep top num_recos
        topk_preds_df['rank'] = (
            topk_preds_df
            .groupby(self.col_user)['score']
            .rank('dense', ascending=False)
            .astype('int64')
        )
        topk_preds_df = (
            topk_preds_df
            .loc[topk_preds_df['rank'] <= self.num_recos]
        )
        return topk_preds_df
