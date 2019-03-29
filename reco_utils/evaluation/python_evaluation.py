# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss
)

from reco_utils.dataset.pandas_df_utils import has_columns, has_same_base_dtype
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    PREDICTION_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)


class Evaluation:

    def __init__(self,
                 df_true,
                 df_pred,
                 col_user,
                 col_item,
                 col_rating,
                 col_prediction):

        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        # check existence of input columns
        if not has_columns(df_true, columns=[col_user, col_item, col_rating]):
            raise ValueError("Truth DataFrame is missing required columns")
        if not has_columns(df_pred, columns=[col_user, col_item, col_prediction]):
            raise ValueError("Prediction DataFrame is missing required columns")

        if not has_same_base_dtype(df_true, df_pred, columns=[col_user, col_item]):
            raise ValueError("Truth and Prediction DataFrame data types are incompatible")

        self.metrics = dict()


class RatingEvaluation(Evaluation):

    def __init__(self,
                 df_true,
                 df_pred,
                 col_user=DEFAULT_USER_COL,
                 col_item=DEFAULT_ITEM_COL,
                 col_rating=DEFAULT_RATING_COL,
                 col_prediction=PREDICTION_COL):
        """Rating prediction evaluation

        Args:
            df_true (pd.DataFrame): truth dataframe
            df_pred (pd.DataFrame): prediction dataframe
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_prediction (str): prediction column name
        """

        super().__init__(df_true=df_true,
                         df_pred=df_pred,
                         col_user=col_user,
                         col_item=col_item,
                         col_rating=col_rating,
                         col_prediction=col_prediction)
        self.y_true, self.y_pred = self.get_true_pred(df_true=df_true, df_pred=df_pred)

    def get_true_pred(self, df_true, df_pred):
        """Join truth and prediction data frames on user and item columns

        Args:
            df_true: truth dataframe
            df_pred: prediction dataframe
        Returns:
            pd.Series, pd.Series: truth data, prediction data
        """

        # select the columns needed for evaluations
        rating_true = df_true[[self.col_user, self.col_item, self.col_rating]]
        rating_pred = df_pred[[self.col_user, self.col_item, self.col_prediction]]

        kwargs = dict()
        if self.col_rating == self.col_prediction:
            kwargs['suffixes'] = ["_true", "_pred"]
            col_true = '{}_true'.format(self.col_rating)
            col_pred = '{}_pred'.format(self.col_prediction)
        else:
            col_true = self.col_rating
            col_pred = self.col_prediction

        df_merged = pd.merge(rating_true,
                             rating_pred,
                             on=[self.col_user, self.col_item],
                             **kwargs)

        return df_merged[col_true], df_merged[col_pred]

    @property
    def auc(self):
        """
        Calculate the Area-Under-Curve metric for implicit feedback typed recommendations,
        where rating is binary and prediction is float number ranging from 0 to 1.

        https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

        Note:
            The evaluation does not require a leave-one-out scenario.
            This metric does not calculate group-based AUC which considers the AUC scores
            averaged across users. It is also not limited to k. Instead, it calculates the
            scores on the entire prediction results regardless the users.

        Returns:
            float: auc_score (min=0, max=1).
        """

        if self.y_true.shape[0] == 0:
            return 0.0

        return roc_auc_score(self.y_true, self.y_pred)

    @property
    def exp_var(self):
        """Calculate explained variance

        Returns:
            float: Explained variance (min=0, max=1).
        """

        if self.y_true.shape[0] == 0:
            return 0.0

        return explained_variance_score(self.y_true, self.y_pred)

    @property
    def logloss(self):
        """
        Calculate the logloss metric for implicit feedback typed recommendations,
        where rating is binary and prediction is float number ranging from 0 to 1.

        https://en.wikipedia.org/wiki/Loss_functions_for_classification#Cross_entropy_loss_(Log_Loss)

        Returns:
            float: log_loss_score (min=-inf, max=inf)
        """

        if self.y_true.shape[0] == 0:
            return np.inf

        return log_loss(self.y_true, self.y_pred)

    @property
    def mae(self):
        """Calculate Mean Absolute Error

        Returns:
            float: Mean Absolute Error
        """

        if self.y_true.shape[0] == 0:
            return np.inf

        return mean_absolute_error(self.y_true, self.y_pred)

    @property
    def rmse(self):
        """Calculate Root Mean Squared Error between rating and prediction columns

        Returns:
            float: Root Mean Squared Error
        """

        if self.y_true.shape[0] == 0:
            return np.inf

        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))

    @property
    def rsquared(self):
        """Calculate R squared

        Returns:
            float: R squared (min=0, max=1).
        """

        if self.y_true.shape[0] == 0:
            return 0.0

        return r2_score(self.y_true, self.y_pred)


class RankingEvaluation(Evaluation):

    def __init__(self,
                 df_true,
                 df_pred,
                 col_user=DEFAULT_USER_COL,
                 col_item=DEFAULT_ITEM_COL,
                 col_rating=DEFAULT_RATING_COL,
                 col_prediction=PREDICTION_COL,
                 relevancy_method='top_k',
                 k=DEFAULT_K,
                 threshold=DEFAULT_THRESHOLD):
        """Ranking prediction evaluation

        Args:
            df_true (pd.DataFrame): truth dataframe
            df_pred (pd.DataFrame): prediction dataframe
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_prediction (str): prediction column name
        """

        super().__init__(df_true=df_true,
                         df_pred=df_pred,
                         col_user=col_user,
                         col_item=col_item,
                         col_rating=col_rating,
                         col_prediction=col_prediction)

        self.relevancy = relevancy_method
        self.k = k
        self.threshold = threshold
        self.col_rank = 'rank'

        common_users = list(set(df_true[col_user]).intersection(set(df_pred[col_user])))
        self.n_users = len(common_users)

        df_pred_common = df_pred[df_pred[col_user].isin(common_users)]
        df_true_common = df_true[df_true[col_user].isin(common_users)]
        self.df_hit = self.get_hits(df_true=df_true_common, df_pred=df_pred_common)
        self.df_hit_count = pd.merge(
            self.df_hit.groupby(col_user).size().reset_index().rename(columns={0: 'hit'}),
            df_true_common.groupby(col_user).size().reset_index().rename(columns={0: 'actual'}),
            on=col_user
        )

    def get_hits(self, df_true, df_pred):
        """Calculate hit items in prediction dataframe with rank information for NDCG and MAP calculation
        This will generate a unique rank for each user-item pair to align the implementation with Spark
        evaluation metrics, where index of each recommended items (the indices are unique to items) is used
        to calculate penalized precision of the ordered items.

        Args:
            df_true (pd.DataFrame): truth dataframe
            df_pred (pd.DataFrame): prediction dataframe
        Returns:
            pd.DataFrame: merged dataframe with recommendation hits
        """

        if self.relevancy == 'top_k':
            df_hit = get_top_k_items(dataframe=df_pred, col_rating=self.col_prediction, k=self.k)
        else:
            raise NotImplementedError

        df_hit["rank"] = (
            df_hit.groupby(self.col_user)[self.col_prediction]
            .rank(method="first", ascending=False)
        )

        df_merge = pd.merge(df_hit, df_true, on=[self.col_user, self.col_item])
        return df_merge[[self.col_user, self.col_item, self.col_rank]]

    @property
    def precision_at_k(self):
        """Precision at K

        Note:
        We use the same formula to calculate precision@k as that in Spark.
        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.precisionAt
        In particular, the maximum achievable precision may be < 1, if the number of items for a
        user in rating_pred is less than k.

        Returns:
            float: precision at k (min=0, max=1)
        """

        if self.df_hit_count.shape[0] == 0:
            return 0.0

        return (self.df_hit_count['hit'] / self.k).mean()

    @property
    def recall_at_k(self):
        """Recall at K
        Maximum value is 1 even when fewer than k items exist for a user in rating_true

        Returns:
            float: recall at k (min=0, max=1)
        """

        if self.df_hit_count.shape[0] == 0:
            return 0.0

        return (self.df_hit_count['hit'] / self.df_hit_count['actual']).mean()

    @property
    def ndcg_at_k(self):
        """Normalized Discounted Cumulative Gain (nDCG)
        Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    
        Returns:
            float: nDCG at k (min=0, max=1)
        """

        if self.df_hit.shape[0] == 0:
            return 0.0

        # calculate discount gain for hit items
        df_dcg = self.df_hit.sort_values(by=[self.col_user, self.col_rank])
        # relevance in this case is always 1
        df_dcg["dcg"] = 1 / np.log1p(df_dcg[self.col_rank])
        # sum up discount gain to get cumulative gain
        df_dcg = df_dcg.groupby(self.col_user).agg({"dcg": "sum"}).reset_index()

        # calculate maximum discounted accumulative gain.
        df_ndcg = pd.merge(df_dcg, self.df_hit_count, on=[self.col_user])
        df_ndcg['idcg'] = df_ndcg['actual'].apply(lambda x: sum(1 / np.log1p(range(1, min(x, self.k) + 1))))

        # DCG over IDCG is the normalized DCG
        return (df_ndcg['dcg'] / df_ndcg['idcg']).mean()

    @property
    def map_at_k(self):
        """
        The implementation of the MAP is referenced from Spark MLlib evaluation metrics.
        https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems

        Get mean average precision at k. A good reference can be found at
        http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

        Note:
            1. The evaluation function is named as 'MAP is at k' because the evaluation class takes top k items for
            the prediction items. The naming is different from Spark.
            2. The MAP is meant to calculate Avg. Precision for the relevant items, so it is normalized by the number of
            relevant items in the ground truth data, instead of k.

        Returns:
            float: MAP at k (min=0, max=1)
        """

        if self.df_hit.shape[0] == 0:
            return 0.0

        # calculate reciprocal rank of items for each user and sum them up
        df_hit = self.df_hit.sort_values([self.col_user, self.col_rank])
        df_hit["rr"] = (df_hit.groupby(self.col_user).cumcount() + 1) / df_hit[self.col_rank]
        df_hit = df_hit.groupby(self.col_user).agg({"rr": "sum"}).reset_index()

        # calculate fraction of sum of reciprocal rank over count of true items
        df_merge = pd.merge(df_hit, self.df_hit_count, on=self.col_user)
        return (df_merge['rr'] / df_merge['actual']).mean()


def get_top_k_items(
    dataframe, col_user=DEFAULT_USER_COL, col_rating=DEFAULT_RATING_COL, k=DEFAULT_K
):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.
    Note:
        if it is implicit rating, just append a column of constants to be
        ratings.

    Args:
        dataframe (pandas.DataFrame): DataFrame of rating data (in the format
        customerID-itemID-rating).
        col_user (str): column name for user.
        col_rating (str): column name for rating.
        k (int): number of items for each user.

    Return:
        pd.DataFrame: DataFrame of top k items for each user.
    """
    return (
        dataframe.groupby(col_user, as_index=False)
        .apply(lambda x: x.nlargest(k, col_rating))
        .reset_index(drop=True)
    )
