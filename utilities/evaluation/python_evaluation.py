"""
PythonEvaluation
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

from utilities.common.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, PREDICTION_COL, \
    DEFAULT_K, DEFAULT_THRESHOLD


class PythonRatingEvaluation:
    """Python Evaluation implementation based on scikit-learn"""

    def __init__(self, rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL,
                 col_rating=DEFAULT_RATING_COL, col_prediction=PREDICTION_COL):
        """Initialization
        Args:
            rating_true (pd.DataFrame): True labels.
            rating_pred (pd.DataFrame): Predicted labels.
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_prediction = col_prediction
        self.col_rating = col_rating
        self.col_user = col_user
        self.col_item = col_item

        if self.col_user not in self.rating_true.columns:
            raise ValueError('Schema of y_true not valid. Missing User Col')
        if self.col_item not in self.rating_true.columns:
            raise ValueError('Schema of y_true not valid. Missing Item Col')
        if self.col_rating not in self.rating_true.columns:
            raise ValueError('Schema of y_true not valid. Missing Rating Col')

        if self.col_user not in self.rating_pred.columns:
            # pragma : No Cover
            raise ValueError('Schema of y_pred not valid. Missing User Col')
        if self.col_item not in self.rating_pred.columns:
            # pragma : No Cover
            raise ValueError('Schema of y_pred not valid. Missing Item Col')
        if self.col_prediction not in self.rating_pred.columns:
            raise ValueError(
                'Schema of y_true not valid. Missing Prediction Col: ' + str(
                    self.rating_pred.columns))

        if col_rating == col_prediction:
            self.rating_true_pred = pd.merge(
                self.rating_true,
                self.rating_pred,
                on=[col_user, col_item],
                suffixes=["_true", "_pred"]
            )

            self.rating_true_pred.rename(columns={col_rating + "_true": DEFAULT_RATING_COL},
                                         inplace=True)
            self.rating_true_pred.rename(columns={col_prediction + "_pred": PREDICTION_COL},
                                         inplace=True)
        else:
            self.rating_true_pred = pd.merge(
                self.rating_true,
                self.rating_pred,
                on=[col_user, col_item]
            )

            self.rating_true_pred.rename(columns={col_rating: DEFAULT_RATING_COL}, inplace=True)
            self.rating_true_pred.rename(columns={col_prediction: PREDICTION_COL}, inplace=True)

    def rmse(self):
        """Calculate Root Mean Squared Error
        Return:
            Root mean squared error.
        """
        return np.sqrt(mean_squared_error(self.rating_true_pred[DEFAULT_RATING_COL],
                                          self.rating_true_pred[PREDICTION_COL]))

    def mae(self):
        """Calculate Mean Absolute Error.
        Returns:
            Mean Absolute Error
        """
        return mean_absolute_error(self.rating_true_pred[DEFAULT_RATING_COL],
                                   self.rating_true_pred[PREDICTION_COL])

    def rsquared(self):
        """Calculate R squared
        Returns:
            R squared
        """
        return r2_score(self.rating_true_pred[DEFAULT_RATING_COL],
                        self.rating_true_pred[PREDICTION_COL])

    def exp_var(self):
        """Calculate explained variance.
        Returns:
            Explained variance
        """
        # return self.metrics.explainedVariance
        return explained_variance_score(self.rating_true_pred[DEFAULT_RATING_COL],
                                        self.rating_true_pred[PREDICTION_COL])


class PythonRankingEvaluation:
    """
    Evaluation with ranking metrics on given data sets.
    """

    def __init__(self, rating_true, rating_pred, k=DEFAULT_K, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL,
                 col_rating=DEFAULT_RATING_COL, col_prediction=PREDICTION_COL, relevancy_method="top_k",
                 threshold=DEFAULT_THRESHOLD):
        """
        Initialization.
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_prediction = col_prediction
        self.col_rating = col_rating
        self.col_user = col_user
        self.col_item = col_item

        self.threshold = threshold
        self.relevancy_method = relevancy_method
        self.top_k = k

        if self.col_user not in self.rating_true.columns:
            raise ValueError('Schema of y_true not valid. Missing User Col')
        if self.col_item not in self.rating_true.columns:
            raise ValueError('Schema of y_true not valid. Missing Item Col')
        if self.col_rating not in self.rating_true.columns:
            raise ValueError('Schema of y_true not valid. Missing Rating Col')

        if self.col_user not in self.rating_pred.columns:
            # pragma : No Cover
            raise ValueError('Schema of y_pred not valid. Missing User Col')
        if self.col_item not in self.rating_pred.columns:
            # pragma : No Cover
            raise ValueError('Schema of y_pred not valid. Missing Item Col')
        if self.col_prediction not in self.rating_pred.columns:
            raise ValueError(
                'Schema of y_pred not valid. Missing Prediction Col: ' + str(
                    self.rating_pred.columns))

        relevant_func = {
            "top_k": get_top_k_items,
        }

        self.rating_pred = relevant_func[relevancy_method](dataframe=self.rating_pred,
                                                           col_user=self.col_user,
                                                           col_item=self.col_item,
                                                           col_rating=self.col_prediction) \
            if relevancy_method == "by_threshold" \
            else relevant_func[relevancy_method](dataframe=self.rating_pred, col_user=self.col_user,
                                                 col_item=self.col_item,
                                                 col_rating=self.col_prediction, k=self.top_k)

        common_user_list = list(
            set(self.rating_true[self.col_user]).intersection(set(self.rating_pred[self.col_user])))

        # To make sure the prediction and true data frames have the same set of users.
        self.rating_pred = self.rating_pred[self.rating_pred[self.col_user].isin(common_user_list)]
        self.rating_true = self.rating_true[self.rating_true[self.col_user].isin(common_user_list)]

        self.df_hit = self._calculate_ranked_hit()
        assert self.df_hit.shape[0] > 0

    def _calculate_ranked_hit(self):
        """Hit data frame between true and pred.

        This function returns hit items in prediction data frame with ranking information. This
        is used for calculating
        NDCG and MAP.
        """

        # Use first to generate unique ranking values for each item. This is to align with the
        # implementation in
        # Spark evaluation metrics, where index of each recommended items (the indices are unique
        #  to items) is used
        # to calculate penalized precision of the ordered items.
        df_rating_pred = self.rating_pred.copy()
        df_rating_pred[self.col_user] = df_rating_pred[self.col_user].astype(int)
        df_rating_pred[self.col_item] = df_rating_pred[self.col_item].astype(int)
        df_rating_pred["ranking"] = self.rating_pred \
            .groupby(self.col_user)[self.col_prediction] \
            .rank(method="first", ascending=False)

        df_hit = \
            pd.merge(self.rating_true, df_rating_pred, how="inner",
                     on=[self.col_user, self.col_item])[
                [self.col_user, self.col_item, "ranking"]]

        return df_hit

    def precision_at_k(self):
        """Precision at K.

        Note we use the same formula to calculate precision@k as that in Spark.
        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation
        .RankingMetrics.precisionAt
        In particular, the maximum achievable precision may be < 1, if the number of items for a
        user
        in rating_pred is less than k.

        Returns:
            result (float): precision at k (max=1, min=0)
        """
        df_count_hit = self.df_hit \
            .groupby(self.col_user) \
            .agg({self.col_item: "count"}) \
            .reset_index() \
            .rename(columns={self.col_item: "hit"}, inplace=False)

        df_count_hit["precision"] = df_count_hit.apply(lambda x: (x.hit / self.top_k), axis=1)

        precision_at_k = np.float64(df_count_hit.agg({"precision": "sum"})) / df_count_hit.shape[0]

        return precision_at_k

    def recall_at_k(self):
        """Recall at K.
        Returns:
            result (float): recall at k (max=1, min=0)
                The maximum value is 1 even when fewer than k items exist for a user in rating_true.
        """
        df_count_hit = self.df_hit \
            .groupby(self.col_user) \
            .agg({self.col_item: "count"}) \
            .reset_index() \
            .rename(columns={self.col_item: "hit"}, inplace=False)

        df_count_true = self.rating_true \
            .groupby(self.col_user) \
            .agg({self.col_item: "count"}) \
            .reset_index() \
            .rename(columns={self.col_item: "actual"}, inplace=False)

        df_count_all = pd.merge(df_count_hit, df_count_true, on=self.col_user)

        df_count_all["recall"] = df_count_all.apply(lambda x: (x.hit / x.actual), axis=1)

        recall_at_k = np.float64(df_count_all.agg({"recall": "sum"})) / df_count_all.shape[0]

        return recall_at_k

    def ndcg_at_k(self):
        """Normalized Discounted Cumulative Gain (nDCG).
        Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
        Returns:
            result (float): nDCG (max=1, min=0)
        """

        # Calculate gain for hit items.
        df_dcg = self.df_hit.sort_values([self.col_user, "ranking"])
        df_dcg["dcg"] = df_dcg.apply(lambda x: 1 / np.log(x.ranking + 1), axis=1)

        # Sum gain up to get accumulative gain.
        df_dcg_sum = df_dcg.groupby(self.col_user).agg({"dcg": "sum"}).reset_index()

        # Helper function to calculate max gain given parameter of n, which is the length of
        # iterations.
        # In calculating the maximum DCG for a user, n is equal to min(number_of_true_items, k).
        def log_sum(iterations_length):
            _sum = 0
            for length in range(iterations_length):
                _sum = _sum + 1 / np.log(length + 2)

            return _sum

        # Calculate maximum discounted accumulative gain.
        df_mdcg_sum = self.rating_true \
            .groupby(self.col_user) \
            .agg({self.col_item: "count"}) \
            .reset_index() \
            .rename(columns={self.col_item: "actual"}, inplace=False)
        df_mdcg_sum["mdcg"] = df_mdcg_sum.apply(lambda x: log_sum(min(x.actual, self.top_k)),
                                                axis=1)

        # DCG over MDCG is the normalized DCG.
        df_ndcg = pd.merge(df_dcg_sum, df_mdcg_sum, on=self.col_user)
        df_ndcg["ndcg"] = df_ndcg.apply(lambda x: x.dcg / x.mdcg, axis=1)

        # Average across users.
        ndcg_at_k = np.float64(df_ndcg.agg({"ndcg": "sum"})) / df_ndcg.shape[0]

        return ndcg_at_k

    def map_at_k(self):
        # pylint: disable=line-too-long
        """
        Get mean average precision at k. A good reference can be found at
        https://people.cs.umass.edu/~jpjiang/cs646/03_eval_basics.pdf

        NOTE the MAP is at k because the evaluation class takes top k items for
        the prediction items.

        Return:
            MAP at k.
        """
        # Calculate inverse of rank of items for each user, use the inverse ranks to penalize
        # precision,
        # and sum them up.
        df_hit = self.df_hit.sort_values([self.col_user, "ranking"])
        df_hit["group_index"] = df_hit.groupby(self.col_user).cumcount() + 1
        df_hit["precision"] = df_hit.apply(lambda x: x.group_index / x.ranking, axis=1)

        df_sum_hit = df_hit.groupby(self.col_user).agg({"precision": "sum"}).reset_index()

        # Count of true items for each user.
        df_count_true = self.rating_true \
            .groupby(self.col_user) \
            .agg({self.col_item: "count"}) \
            .reset_index() \
            .rename(columns={self.col_item: "actual"}, inplace=False)

        # Calculate proportion of sum of inverse rank over count of true items.
        df_sum_all = pd.merge(df_sum_hit, df_count_true, on=self.col_user)
        df_sum_all["map"] = df_sum_all.apply(lambda x: (x.precision / x.actual), axis=1)

        # Average the results across users.
        map_at_k = np.float64(df_sum_all.agg({"map": "sum"})) / df_sum_all.shape[0]

        return map_at_k


def get_top_k_items(dataframe, col_user="customerID", col_item="itemID",
                    col_rating="rating", k=10):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.
    Note:
        if it is implicit rating, just append a column of constants to be
        ratings.

    Args:
        dataframe (pandas.DataFrame): DataFrame of rating data (in the format of
        customerID-itemID-rating tuple).
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        k (int): number of items for each user.

    Return:
        Pandas DataFrame of top k items for each user.
    """
    dataframe[col_rating] = dataframe[col_rating].astype(float)
    return dataframe \
        .groupby(col_user, as_index=False) \
        .apply(lambda x: x.nlargest(k, col_rating)) \
        .reset_index()[dataframe.columns]
