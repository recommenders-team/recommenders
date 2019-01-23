# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    PREDICTION_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)


def _merge_rating_true_pred(
    rating_true, rating_pred, col_user, col_item, col_rating, col_prediction
):
    """Join truth and prediction data frames on userID and itemID
    
    Args:
        rating_true (pd.DataFrame): True data.
        rating_pred (pd.DataFrame): Predicted data.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.

    Returns:
        pd.DataFrame: Merged pd.DataFrame
    """

    if col_user not in rating_true.columns:
        raise ValueError("Schema of y_true not valid. Missing User Col")
    if col_item not in rating_true.columns:
        raise ValueError("Schema of y_true not valid. Missing Item Col")
    if col_rating not in rating_true.columns:
        raise ValueError("Schema of y_true not valid. Missing Rating Col")

    if col_user not in rating_pred.columns:
        # pragma : No Cover
        raise ValueError("Schema of y_pred not valid. Missing User Col")
    if col_item not in rating_pred.columns:
        # pragma : No Cover
        raise ValueError("Schema of y_pred not valid. Missing Item Col")
    if col_prediction not in rating_pred.columns:
        raise ValueError(
            "Schema of y_true not valid. Missing Prediction Col: "
            + str(rating_pred.columns)
        )

    # Select the columns needed for evaluations
    rating_true = rating_true[[col_user, col_item, col_rating]]
    rating_pred = rating_pred[[col_user, col_item, col_prediction]]

    if col_rating == col_prediction:
        rating_true_pred = pd.merge(
            rating_true,
            rating_pred,
            on=[col_user, col_item],
            suffixes=["_true", "_pred"],
        )
        rating_true_pred.rename(
            columns={col_rating + "_true": DEFAULT_RATING_COL}, inplace=True
        )
        rating_true_pred.rename(
            columns={col_prediction + "_pred": PREDICTION_COL}, inplace=True
        )
    else:
        rating_true_pred = pd.merge(rating_true, rating_pred, on=[col_user, col_item])
        rating_true_pred.rename(columns={col_rating: DEFAULT_RATING_COL}, inplace=True)
        rating_true_pred.rename(columns={col_prediction: PREDICTION_COL}, inplace=True)

    return rating_true_pred


def rmse(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
):
    """Calculate Root Mean Squared Error

    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs.
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
    
    Returns:
        float: Root mean squared error.
    """
    rating_true_pred = _merge_rating_true_pred(
        rating_true, rating_pred, col_user, col_item, col_rating, col_prediction
    )
    return np.sqrt(
        mean_squared_error(
            rating_true_pred[DEFAULT_RATING_COL], rating_true_pred[PREDICTION_COL]
        )
    )


def mae(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
):
    """Calculate Mean Absolute Error.

    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs.
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.

    Returns:
        float: Mean Absolute Error.
    """
    rating_true_pred = _merge_rating_true_pred(
        rating_true, rating_pred, col_user, col_item, col_rating, col_prediction
    )
    return mean_absolute_error(
        rating_true_pred[DEFAULT_RATING_COL], rating_true_pred[PREDICTION_COL]
    )


def rsquared(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
):
    """Calculate R squared

    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs.
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
    
    Returns:
        float: R squared (min=0, max=1).
    """
    rating_true_pred = _merge_rating_true_pred(
        rating_true, rating_pred, col_user, col_item, col_rating, col_prediction
    )
    return r2_score(
        rating_true_pred[DEFAULT_RATING_COL], rating_true_pred[PREDICTION_COL]
    )


def exp_var(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
):
    """Calculate explained variance.

    Args:
        rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs.
        rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.

    Returns:
        float: Explained variance (min=0, max=1).
    """
    rating_true_pred = _merge_rating_true_pred(
        rating_true, rating_pred, col_user, col_item, col_rating, col_prediction
    )
    return explained_variance_score(
        rating_true_pred[DEFAULT_RATING_COL], rating_true_pred[PREDICTION_COL]
    )


def _merge_ranking_true_pred(
    rating_true,
    rating_pred,
    col_user,
    col_item,
    col_rating,
    col_prediction,
    relevancy_method,
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Filter truth and prediction data frames on common users

    Args:
        rating_true (pd.DataFrame): True data.
        rating_pred (pd.DataFrame): Predicted data.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.

    Returns:
        pd.DataFrame: new data frame of true data DataFrame of recommendation hits
            number of common users
    """

    if col_user not in rating_true.columns:
        raise ValueError("Schema of y_true not valid. Missing User Col")
    if col_item not in rating_true.columns:
        raise ValueError("Schema of y_true not valid. Missing Item Col")
    if col_rating not in rating_true.columns:
        raise ValueError("Schema of y_true not valid. Missing Rating Col")

    if col_user not in rating_pred.columns:
        # pragma : No Cover
        raise ValueError("Schema of y_pred not valid. Missing User Col")
    if col_item not in rating_pred.columns:
        # pragma : No Cover
        raise ValueError("Schema of y_pred not valid. Missing Item Col")
    if col_prediction not in rating_pred.columns:
        raise ValueError(
            "Schema of y_pred not valid. Missing Prediction Col: "
            + str(rating_pred.columns)
        )

    relevant_func = {"top_k": get_top_k_items}

    rating_pred_new = (
        relevant_func[relevancy_method](
            dataframe=rating_pred,
            col_user=col_user,
            col_rating=col_prediction,
            threshold=threshold,
        )
        if relevancy_method == "by_threshold"
        else relevant_func[relevancy_method](
            dataframe=rating_pred, col_user=col_user, col_rating=col_prediction, k=k
        )
    )

    common_user_list = list(
        set(rating_true[col_user]).intersection(set(rating_pred_new[col_user]))
    )

    # To make sure the prediction and true data frames have the same set of users.
    rating_pred_new = rating_pred_new[rating_pred_new[col_user].isin(common_user_list)]
    rating_true_new = rating_true[rating_true[col_user].isin(common_user_list)]

    n_users = len(common_user_list)

    # Return hit items in prediction data frame with ranking information. This
    # is used for calculating
    # NDCG and MAP.
    # Use first to generate unique ranking values for each item. This is to align with the
    # implementation in
    # Spark evaluation metrics, where index of each recommended items (the indices are unique
    #  to items) is used
    # to calculate penalized precision of the ordered items.
    df_rating_pred = rating_pred_new.copy()
    df_rating_pred["ranking"] = rating_pred_new.groupby(col_user)[col_prediction].rank(
        method="first", ascending=False
    )

    df_hit = pd.merge(
        rating_true_new, df_rating_pred, how="inner", on=[col_user, col_item]
    )[[col_user, col_item, "ranking"]]

    return rating_true_new, df_hit, n_users


def precision_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Precision at K.

    Note:
    We use the same formula to calculate precision@k as that in Spark.
    More details can be found at
    http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.precisionAt
    In particular, the maximum achievable precision may be < 1, if the number of items for a
    user in rating_pred is less than k.

    Args:
        rating_true (pd.DataFrame): True data.
        rating_pred (pd.DataFrame): Predicted data.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        relevancy_method (str): method for getting the most relevant items.
        k (int): number of top k items per user.
        threshold (float): threshold of top items per user (optional).
    
    Returns:
        float: precision at k (min=0, max=1)
    """
    _, df_hit, n_users = _merge_ranking_true_pred(
        rating_true,
        rating_pred,
        col_user,
        col_item,
        col_rating,
        col_prediction,
        relevancy_method,
        k,
        threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_count_hit = (
        df_hit.groupby(col_user)
        .agg({col_item: "count"})
        .reset_index()
        .rename(columns={col_item: "hit"}, inplace=False)
    )

    df_count_hit["precision"] = df_count_hit.apply(lambda x: (x.hit / k), axis=1)

    return np.float64(df_count_hit.agg({"precision": "sum"})) / n_users


def recall_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Recall at K.

    Args:
        rating_true (pd.DataFrame): True data.
        rating_pred (pd.DataFrame): Predicted data.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        relevancy_method (str): method for getting the most relevant items.
        k (int): number of top k items per user.
        threshold (float): threshold of top items per user (optional).
    
    Returns:
        float: recall at k (min=0, max=1). The maximum value is 1 even when fewer than 
            k items exist for a user in rating_true.
    """
    rating_true_new, df_hit, n_users = _merge_ranking_true_pred(
        rating_true,
        rating_pred,
        col_user,
        col_item,
        col_rating,
        col_prediction,
        relevancy_method,
        k,
        threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    df_count_hit = (
        df_hit.groupby(col_user)
        .agg({col_item: "count"})
        .reset_index()
        .rename(columns={col_item: "hit"}, inplace=False)
    )

    df_count_true = (
        rating_true_new.groupby(col_user)
        .agg({col_item: "count"})
        .reset_index()
        .rename(columns={col_item: "actual"}, inplace=False)
    )

    df_count_all = pd.merge(df_count_hit, df_count_true, on=col_user)

    df_count_all["recall"] = df_count_all.apply(lambda x: (x.hit / x.actual), axis=1)

    return np.float64(df_count_all.agg({"recall": "sum"})) / n_users


def ndcg_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Normalized Discounted Cumulative Gain (nDCG).
    
    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    
    Args:
        rating_true (pd.DataFrame): True data.
        rating_pred (pd.DataFrame): Predicted data.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        relevancy_method (str): method for getting the most relevant items.
        k (int): number of top k items per user.
        threshold (float): threshold of top items per user (optional).
    
    Returns:
        float: nDCG at k (min=0, max=1).
    """
    rating_true_new, df_hit, n_users = _merge_ranking_true_pred(
        rating_true,
        rating_pred,
        col_user,
        col_item,
        col_rating,
        col_prediction,
        relevancy_method,
        k,
        threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    # Calculate gain for hit items.
    df_dcg = df_hit.sort_values([col_user, "ranking"])
    df_dcg["dcg"] = df_dcg.apply(lambda x: 1 / np.log(x.ranking + 1), axis=1)

    # Sum gain up to get accumulative gain.
    df_dcg_sum = df_dcg.groupby(col_user).agg({"dcg": "sum"}).reset_index()

    # Helper function to calculate max gain given parameter of n, which is the length of
    # iterations.
    # In calculating the maximum DCG for a user, n is equal to min(number_of_true_items, k).
    def log_sum(iterations_length):
        _sum = 0
        for length in range(iterations_length):
            _sum = _sum + 1 / np.log(length + 2)

        return _sum

    # Calculate maximum discounted accumulative gain.
    df_mdcg_sum = (
        rating_true_new.groupby(col_user)
        .agg({col_item: "count"})
        .reset_index()
        .rename(columns={col_item: "actual"}, inplace=False)
    )
    df_mdcg_sum["mdcg"] = df_mdcg_sum.apply(lambda x: log_sum(min(x.actual, k)), axis=1)

    # DCG over MDCG is the normalized DCG.
    df_ndcg = pd.merge(df_dcg_sum, df_mdcg_sum, on=col_user)
    df_ndcg["ndcg"] = df_ndcg.apply(lambda x: x.dcg / x.mdcg, axis=1)

    # Average across users.
    return np.float64(df_ndcg.agg({"ndcg": "sum"})) / n_users


def map_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Get mean average precision at k. A good reference can be found at
    https://people.cs.umass.edu/~jpjiang/cs646/03_eval_basics.pdf

    NOTE: The MAP is at k because the evaluation class takes top k items for
    the prediction items.

    Args:
        rating_true (pd.DataFrame): True data.
        rating_pred (pd.DataFrame): Predicted data.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        relevancy_method (str): method for getting the most relevant items.
        k (int): number of top k items per user.
        threshold (float): threshold of top items per user (optional).
    
    Return:
        float: MAP at k (min=0, max=1).
    """
    rating_true_new, df_hit, n_users = _merge_ranking_true_pred(
        rating_true,
        rating_pred,
        col_user,
        col_item,
        col_rating,
        col_prediction,
        relevancy_method,
        k,
        threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    # Calculate inverse of rank of items for each user, use the inverse ranks to penalize
    # precision,
    # and sum them up.
    df_hit = df_hit.sort_values([col_user, "ranking"])
    df_hit["group_index"] = df_hit.groupby(col_user).cumcount() + 1
    df_hit["precision"] = df_hit.apply(lambda x: x.group_index / x.ranking, axis=1)

    df_sum_hit = df_hit.groupby(col_user).agg({"precision": "sum"}).reset_index()

    # Count of true items for each user.
    df_count_true = (
        rating_true_new.groupby(col_user)
        .agg({col_item: "count"})
        .reset_index()
        .rename(columns={col_item: "actual"}, inplace=False)
    )

    # Calculate proportion of sum of inverse rank over count of true items.
    df_sum_all = pd.merge(df_sum_hit, df_count_true, on=col_user)
    df_sum_all["map"] = df_sum_all.apply(lambda x: (x.precision / x.actual), axis=1)

    # Average the results across users.
    return np.float64(df_sum_all.agg({"map": "sum"})) / n_users


def get_top_k_items(dataframe, col_user="customerID", col_rating="rating", k=10):
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
    dataframe.loc[:, col_rating] = dataframe[col_rating].astype(float)
    return (
        dataframe.groupby(col_user, as_index=False)
        .apply(lambda x: x.nlargest(k, col_rating))
        .reset_index()[dataframe.columns]
    )
