# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from functools import wraps
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
    log_loss,
)

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RELEVANCE_COL,
    DEFAULT_SIMILARITY_COL,
    DEFAULT_ITEM_FEATURES_COL,
    DEFAULT_ITEM_SIM_MEASURE,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
)
from recommenders.datasets.pandas_df_utils import (
    has_columns,
    has_same_base_dtype,
    lru_cache_df,
)


def check_column_dtypes(func):
    """Checks columns of DataFrame inputs

    This includes the checks on:

    * whether the input columns exist in the input DataFrames
    * whether the data types of col_user as well as col_item are matched in the two input DataFrames.

    Args:
        func (function): function that will be wrapped

    Returns:
        function: Wrapper function for checking dtypes.
    """

    @wraps(func)
    def check_column_dtypes_wrapper(
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        *args,
        **kwargs
    ):
        """Check columns of DataFrame inputs

        Args:
            rating_true (pandas.DataFrame): True data
            rating_pred (pandas.DataFrame): Predicted data
            col_user (str): column name for user
            col_item (str): column name for item
            col_rating (str): column name for rating
            col_prediction (str): column name for prediction
        """

        if not has_columns(rating_true, [col_user, col_item, col_rating]):
            raise ValueError("Missing columns in true rating DataFrame")
        if not has_columns(rating_pred, [col_user, col_item, col_prediction]):
            raise ValueError("Missing columns in predicted rating DataFrame")
        if not has_same_base_dtype(
            rating_true, rating_pred, columns=[col_user, col_item]
        ):
            raise ValueError("Columns in provided DataFrames are not the same datatype")

        return func(
            rating_true=rating_true,
            rating_pred=rating_pred,
            col_user=col_user,
            col_item=col_item,
            col_rating=col_rating,
            col_prediction=col_prediction,
            *args,
            **kwargs
        )

    return check_column_dtypes_wrapper


@check_column_dtypes
@lru_cache_df(maxsize=1)
def merge_rating_true_pred(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Join truth and prediction data frames on userID and itemID and return the true
    and predicted rated with the correct index.

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        numpy.ndarray: Array with the true ratings
        numpy.ndarray: Array with the predicted ratings

    """

    # pd.merge will apply suffixes to columns which have the same name across both dataframes
    suffixes = ["_true", "_pred"]
    rating_true_pred = pd.merge(
        rating_true, rating_pred, on=[col_user, col_item], suffixes=suffixes
    )
    if col_rating in rating_pred.columns:
        col_rating = col_rating + suffixes[0]
    if col_prediction in rating_true.columns:
        col_prediction = col_prediction + suffixes[1]
    return rating_true_pred[col_rating], rating_true_pred[col_prediction]


def rmse(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate Root Mean Squared Error

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: Root mean squared error
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate Mean Absolute Error.

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: Mean Absolute Error.
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return mean_absolute_error(y_true, y_pred)


def rsquared(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate R squared

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: R squared (min=0, max=1).
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return r2_score(y_true, y_pred)


def exp_var(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate explained variance.

    Args:
        rating_true (pandas.DataFrame): True data. There should be no duplicate (userID, itemID) pairs
        rating_pred (pandas.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: Explained variance (min=0, max=1).
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return explained_variance_score(y_true, y_pred)


def auc(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate the Area-Under-Curve metric for implicit feedback typed
    recommender, where rating is binary and prediction is float number ranging
    from 0 to 1.

    https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve

    Note:
        The evaluation does not require a leave-one-out scenario.
        This metric does not calculate group-based AUC which considers the AUC scores
        averaged across users. It is also not limited to k. Instead, it calculates the
        scores on the entire prediction results regardless the users.

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: auc_score (min=0, max=1)
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return roc_auc_score(y_true, y_pred)


def logloss(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Calculate the logloss metric for implicit feedback typed
    recommender, where rating is binary and prediction is float number ranging
    from 0 to 1.

    https://en.wikipedia.org/wiki/Loss_functions_for_classification#Cross_entropy_loss_(Log_Loss)

    Args:
        rating_true (pandas.DataFrame): True data
        rating_pred (pandas.DataFrame): Predicted data
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction

    Returns:
        float: log_loss_score (min=-inf, max=inf)
    """

    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return log_loss(y_true, y_pred)


@check_column_dtypes
@lru_cache_df(maxsize=1)
def merge_ranking_true_pred(
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
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user (optional)
        threshold (float): threshold of top items per user (optional)

    Returns:
        pandas.DataFrame, pandas.DataFrame, int: DataFrame of recommendation hits, sorted by `col_user` and `rank`
        DataFrmae of hit counts vs actual relevant items per user number of unique user ids
    """

    # Make sure the prediction and true data frames have the same set of users
    common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user]))
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    n_users = len(common_users)

    # Return hit items in prediction data frame with ranking information. This is used for calculating NDCG and MAP.
    # Use first to generate unique ranking values for each item. This is to align with the implementation in
    # Spark evaluation metrics, where index of each recommended items (the indices are unique to items) is used
    # to calculate penalized precision of the ordered items.
    if relevancy_method == "top_k":
        top_k = k
    elif relevancy_method == "by_threshold":
        top_k = threshold
    elif relevancy_method is None:
        top_k = None
    else:
        raise NotImplementedError("Invalid relevancy_method")
    df_hit = get_top_k_items(
        dataframe=rating_pred_common,
        col_user=col_user,
        col_rating=col_prediction,
        k=top_k,
    )
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]

    # count the number of hits vs actual relevant items per user
    df_hit_count = pd.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
    )

    return df_hit, df_hit_count, n_users


def precision_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
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
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)

    Returns:
        float: precision at k (min=0, max=1)
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / k).sum() / n_users


def recall_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Recall at K.

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)

    Returns:
        float: recall at k (min=0, max=1). The maximum value is 1 even when fewer than
        k items exist for a user in rating_true.
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users


def ndcg_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Normalized Discounted Cumulative Gain (nDCG).

    Info: https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)

    Returns:
        float: nDCG at k (min=0, max=1).
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    # calculate discounted gain for hit items
    df_dcg = df_hit.copy()
    # relevance in this case is always 1
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])
    # sum up discount gained to get discount cumulative gain
    df_dcg = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})
    # calculate ideal discounted cumulative gain
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=[col_user])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )

    # DCG over IDCG is the normalized DCG
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users


def map_at_k(
    rating_true,
    rating_pred,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    relevancy_method="top_k",
    k=DEFAULT_K,
    threshold=DEFAULT_THRESHOLD,
):
    """Mean Average Precision at k

    The implementation of MAP is referenced from Spark MLlib evaluation metrics.
    https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems

    A good reference can be found at:
    http://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf

    Note:
        1. The evaluation function is named as 'MAP is at k' because the evaluation class takes top k items for
        the prediction items. The naming is different from Spark.

        2. The MAP is meant to calculate Avg. Precision for the relevant items, so it is normalized by the number of
        relevant items in the ground truth data, instead of k.

    Args:
        rating_true (pandas.DataFrame): True DataFrame
        rating_pred (pandas.DataFrame): Predicted DataFrame
        col_user (str): column name for user
        col_item (str): column name for item
        col_rating (str): column name for rating
        col_prediction (str): column name for prediction
        relevancy_method (str): method for determining relevancy ['top_k', 'by_threshold', None]. None means that the
            top k items are directly provided, so there is no need to compute the relevancy operation.
        k (int): number of top k items per user
        threshold (float): threshold of top items per user (optional)

    Returns:
        float: MAP at k (min=0, max=1).
    """

    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )

    if df_hit.shape[0] == 0:
        return 0.0

    # calculate reciprocal rank of items for each user and sum them up
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rr"] = (
        df_hit_sorted.groupby(col_user).cumcount() + 1
    ) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby(col_user).agg({"rr": "sum"}).reset_index()

    df_merge = pd.merge(df_hit_sorted, df_hit_count, on=col_user)
    return (df_merge["rr"] / df_merge["actual"]).sum() / n_users


def get_top_k_items(
    dataframe, col_user=DEFAULT_USER_COL, col_rating=DEFAULT_RATING_COL, k=DEFAULT_K
):
    """Get the input customer-item-rating tuple in the format of Pandas
    DataFrame, output a Pandas DataFrame in the dense format of top k items
    for each user.

    Note:
        If it is implicit rating, just append a column of constants to be
        ratings.

    Args:
        dataframe (pandas.DataFrame): DataFrame of rating data (in the format
        customerID-itemID-rating)
        col_user (str): column name for user
        col_rating (str): column name for rating
        k (int or None): number of items for each user; None means that the input has already been
        filtered out top k items and sorted by ratings and there is no need to do that again.

    Returns:
        pandas.DataFrame: DataFrame of top k items for each user, sorted by `col_user` and `rank`
    """
    # Sort dataframe by col_user and (top k) col_rating
    if k is None:
        top_k_items = dataframe
    else:
        top_k_items = (
            dataframe.groupby(col_user, as_index=False)
            .apply(lambda x: x.nlargest(k, col_rating))
            .reset_index(drop=True)
        )
    # Add ranks
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names"""
metrics = {
    rmse.__name__: rmse,
    mae.__name__: mae,
    rsquared.__name__: rsquared,
    exp_var.__name__: exp_var,
    precision_at_k.__name__: precision_at_k,
    recall_at_k.__name__: recall_at_k,
    ndcg_at_k.__name__: ndcg_at_k,
    map_at_k.__name__: map_at_k,
}

# diversity metrics
def check_column_dtypes_diversity_serendipity(func):
    """Checks columns of DataFrame inputs

    This includes the checks on:

    * whether the input columns exist in the input DataFrames
    * whether the data types of col_user as well as col_item are matched in the two input DataFrames.
    * whether reco_df contains any user_item pairs that are already shown in train_df
    * check relevance column in reco_df
    * check column names in item_feature_df

    Args:
        func (function): function that will be wrapped

    Returns:
        function: Wrapper function for checking dtypes.
    """

    @wraps(func)
    def check_column_dtypes_diversity_serendipity_wrapper(
        train_df,
        reco_df,
        item_feature_df=None,
        item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
        col_item_features=DEFAULT_ITEM_FEATURES_COL,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_sim=DEFAULT_SIMILARITY_COL,
        col_relevance=None,
        *args,
        **kwargs
    ):
        """Check columns of DataFrame inputs

        Args:
          train_df (pandas.DataFrame): Data set with historical data for users and items they
              have interacted with; contains col_user, col_item. Assumed to not contain any duplicate rows.
          reco_df (pandas.DataFrame): Recommender's prediction output, containing col_user, col_item,
              col_relevance (optional). Assumed to not contain any duplicate user-item pairs.
          item_feature_df (pandas.DataFrame): (Optional) It is required only when item_sim_measure='item_feature_vector'. It contains two columns: col_item and features (a feature vector).
          item_sim_measure (str): (Optional) This column indicates which item similarity measure to be used. Available measures include item_cooccurrence_count (default choice) and item_feature_vector.
          col_item_features (str): item feature column name.
          col_user (str): User id column name.
          col_item (str): Item id column name.
          col_relevance (str): This column indicates whether the recommended item is actually
              relevant to the user or not.
        """

        if not has_columns(train_df, [col_user, col_item]):
            raise ValueError("Missing columns in train_df DataFrame")
        if not has_columns(reco_df, [col_user, col_item]):
            raise ValueError("Missing columns in reco_df DataFrame")
        if not has_same_base_dtype(train_df, reco_df, columns=[col_user, col_item]):
            raise ValueError("Columns in provided DataFrames are not the same datatype")
        if col_relevance is None:
            col_relevance = DEFAULT_RELEVANCE_COL
            # relevance term, default is 1 (relevant) for all
            reco_df = reco_df[[col_user, col_item]]
            reco_df[col_relevance] = 1.0
        else:
            col_relevance = col_relevance
            reco_df = reco_df[[col_user, col_item, col_relevance]].astype(
                {col_relevance: np.float16}
            )
        if item_sim_measure == "item_feature_vector":
            required_columns = [col_item, col_item_features]
            if item_feature_df is not None:
                if not has_columns(item_feature_df, required_columns):
                    raise ValueError("Missing columns in item_feature_df DataFrame")
            else:
                raise Exception(
                    "item_feature_df not specified! item_feature_df must be provided if choosing to use item_feature_vector to calculate item similarity. item_feature_df should have columns:"
                    + str(required_columns)
                )
        # check if reco_df contains any user_item pairs that are already shown in train_df
        count_intersection = pd.merge(
            train_df, reco_df, how="inner", on=[col_user, col_item]
        ).shape[0]
        if count_intersection != 0:
            raise Exception(
                "reco_df should not contain any user_item pairs that are already shown in train_df"
            )

        return func(
            train_df=train_df,
            reco_df=reco_df,
            item_feature_df=item_feature_df,
            item_sim_measure=item_sim_measure,
            col_user=col_user,
            col_item=col_item,
            col_sim=col_sim,
            col_relevance=col_relevance,
            *args,
            **kwargs
        )

    return check_column_dtypes_diversity_serendipity_wrapper


def check_column_dtypes_novelty_coverage(func):
    """Checks columns of DataFrame inputs

    This includes the checks on:

    * whether the input columns exist in the input DataFrames
    * whether the data types of col_user as well as col_item are matched in the two input DataFrames.
    * whether reco_df contains any user_item pairs that are already shown in train_df

    Args:
        func (function): function that will be wrapped

    Returns:
        function: Wrapper function for checking dtypes.
    """

    @wraps(func)
    def check_column_dtypes_novelty_coverage_wrapper(
        train_df,
        reco_df,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        *args,
        **kwargs
    ):
        """Check columns of DataFrame inputs

        Args:
            train_df (pandas.DataFrame): Data set with historical data for users and items they
                have interacted with; contains col_user, col_item. Assumed to not contain any duplicate rows.
                Interaction here follows the *item choice model* from Castells et al.
            reco_df (pandas.DataFrame): Recommender's prediction output, containing col_user, col_item,
                col_relevance (optional). Assumed to not contain any duplicate user-item pairs.
            col_user (str): User id column name.
            col_item (str): Item id column name.

        """

        if not has_columns(train_df, [col_user, col_item]):
            raise ValueError("Missing columns in train_df DataFrame")
        if not has_columns(reco_df, [col_user, col_item]):
            raise ValueError("Missing columns in reco_df DataFrame")
        if not has_same_base_dtype(train_df, reco_df, columns=[col_user, col_item]):
            raise ValueError("Columns in provided DataFrames are not the same datatype")

        count_intersection = pd.merge(
            train_df, reco_df, how="inner", on=[col_user, col_item]
        ).shape[0]
        if count_intersection != 0:
            raise Exception(
                "reco_df should not contain any user_item pairs that are already shown in train_df"
            )

        return func(
            train_df=train_df,
            reco_df=reco_df,
            col_user=col_user,
            col_item=col_item,
            *args,
            **kwargs
        )

    return check_column_dtypes_novelty_coverage_wrapper


@lru_cache_df(maxsize=1)
def _get_pairwise_items(
    df,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Get pairwise combinations of items per user (ignoring duplicate pairs [1,2] == [2,1])"""
    df_user_i1 = df[[col_user, col_item]]
    df_user_i1.columns = [col_user, "i1"]

    df_user_i2 = df[[col_user, col_item]]
    df_user_i2.columns = [col_user, "i2"]

    df_user_i1_i2 = pd.merge(df_user_i1, df_user_i2, how="inner", on=[col_user])

    df_pairwise_items = df_user_i1_i2[(df_user_i1_i2["i1"] <= df_user_i1_i2["i2"])][
        [col_user, "i1", "i2"]
    ].reset_index(drop=True)
    return df_pairwise_items


@lru_cache_df(maxsize=1)
def _get_cosine_similarity(
    train_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
):

    if item_sim_measure == "item_cooccurrence_count":
        # calculate item-item similarity based on item co-occurrence count
        df_cosine_similarity = _get_cooccurrence_similarity(
            train_df, col_user, col_item, col_sim
        )
    elif item_sim_measure == "item_feature_vector":
        # calculdf_cosine_similarity = ate item-item similarity based on item feature vectors
        df_cosine_similarity = _get_item_feature_similarity(
            item_feature_df, col_item_features, col_user, col_item
        )
    else:
        raise Exception(
            "item_sim_measure not recognized! The available options include 'item_cooccurrence_count' and 'item_feature_vector'."
        )
    return df_cosine_similarity


@lru_cache_df(maxsize=1)
def _get_cooccurrence_similarity(
    train_df,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
):
    """Cosine similarity metric from

    :Citation:

        Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
        introducing serendipity into music recommendation, WSDM 2012

    The item indexes in the result are such that i1 <= i2.
    """
    pairs = _get_pairwise_items(train_df, col_user, col_item)
    pairs_count = pd.DataFrame(
        {"count": pairs.groupby(["i1", "i2"]).size()}
    ).reset_index()
    item_count = pd.DataFrame(
        {"count": train_df.groupby([col_item]).size()}
    ).reset_index()
    item_count["item_sqrt_count"] = item_count["count"] ** 0.5
    item_co_occur = pairs_count.merge(
        item_count[[col_item, "item_sqrt_count"]],
        left_on=["i1"],
        right_on=[col_item],
    ).drop(columns=[col_item])

    item_co_occur.columns = ["i1", "i2", "count", "i1_sqrt_count"]

    item_co_occur = item_co_occur.merge(
        item_count[[col_item, "item_sqrt_count"]],
        left_on=["i2"],
        right_on=[col_item],
    ).drop(columns=[col_item])
    item_co_occur.columns = [
        "i1",
        "i2",
        "count",
        "i1_sqrt_count",
        "i2_sqrt_count",
    ]

    item_co_occur[col_sim] = item_co_occur["count"] / (
        item_co_occur["i1_sqrt_count"] * item_co_occur["i2_sqrt_count"]
    )
    df_cosine_similarity = (
        item_co_occur[["i1", "i2", col_sim]]
        .sort_values(["i1", "i2"])
        .reset_index(drop=True)
    )

    return df_cosine_similarity


@lru_cache_df(maxsize=1)
def _get_item_feature_similarity(
    item_feature_df,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
):
    """Cosine similarity metric based on item feature vectors

    The item indexes in the result are such that i1 <= i2.
    """
    df1 = item_feature_df[[col_item, col_item_features]]
    df1.columns = ["i1", "f1"]
    df1["key"] = 0
    df2 = item_feature_df[[col_item, col_item_features]]
    df2.columns = ["i2", "f2"]
    df2["key"] = 0

    df = pd.merge(df1, df2, on="key", how="outer").drop("key", axis=1)
    df_item_feature_pair = df[(df["i1"] <= df["i2"])].reset_index(drop=True)

    df_item_feature_pair[col_sim] = df_item_feature_pair.apply(
        lambda x: float(x.f1.dot(x.f2))
        / float(np.linalg.norm(x.f1, 2) * np.linalg.norm(x.f2, 2)),
        axis=1,
    )

    df_cosine_similarity = df_item_feature_pair[["i1", "i2", col_sim]].sort_values(
        ["i1", "i2"]
    )

    return df_cosine_similarity


# Diversity metrics
@lru_cache_df(maxsize=1)
def _get_intralist_similarity(
    train_df,
    reco_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
):
    """Intra-list similarity from

    :Citation:

        "Improving Recommendation Lists Through Topic Diversification",
        Ziegler, McNee, Konstan and Lausen, 2005.
    """
    pairs = _get_pairwise_items(reco_df, col_user, col_item)
    similarity_df = _get_cosine_similarity(
        train_df,
        item_feature_df,
        item_sim_measure,
        col_item_features,
        col_user,
        col_item,
        col_sim,
    )
    # Fillna(0) is needed in the cases where similarity_df does not have an entry for a pair of items.
    # e.g. i1 and i2 have never occurred together.

    item_pair_sim = pairs.merge(similarity_df, on=["i1", "i2"], how="left")
    item_pair_sim[col_sim].fillna(0, inplace=True)
    item_pair_sim = item_pair_sim.loc[
        item_pair_sim["i1"] != item_pair_sim["i2"]
    ].reset_index(drop=True)
    df_intralist_similarity = (
        item_pair_sim.groupby([col_user]).agg({col_sim: "mean"}).reset_index()
    )
    df_intralist_similarity.columns = [col_user, "avg_il_sim"]

    return df_intralist_similarity


@check_column_dtypes_diversity_serendipity
@lru_cache_df(maxsize=1)
def user_diversity(
    train_df,
    reco_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
    col_relevance=None,
):
    """Calculate average diversity of recommendations for each user.
    The metric definition is based on formula (3) in the following reference:

    :Citation:

        Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
        introducing serendipity into music recommendation, WSDM 2012

    Returns:
        pandas.DataFrame: A dataframe with the following columns: col_user, user_diversity.
    """

    df_intralist_similarity = _get_intralist_similarity(
        train_df,
        reco_df,
        item_feature_df,
        item_sim_measure,
        col_item_features,
        col_user,
        col_item,
        col_sim,
    )
    df_user_diversity = df_intralist_similarity
    df_user_diversity["user_diversity"] = 1 - df_user_diversity["avg_il_sim"]
    df_user_diversity = (
        df_user_diversity[[col_user, "user_diversity"]]
        .sort_values(col_user)
        .reset_index(drop=True)
    )

    return df_user_diversity


@check_column_dtypes_diversity_serendipity
def diversity(
    train_df,
    reco_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
    col_relevance=None,
):
    """Calculate average diversity of recommendations across all users.

    Returns:
        float: diversity.
    """
    df_user_diversity = user_diversity(
        train_df,
        reco_df,
        item_feature_df,
        item_sim_measure,
        col_item_features,
        col_user,
        col_item,
        col_sim,
    )
    avg_diversity = df_user_diversity.agg({"user_diversity": "mean"})[0]
    return avg_diversity


# Novelty metrics
@check_column_dtypes_novelty_coverage
@lru_cache_df(maxsize=1)
def historical_item_novelty(
    train_df,
    reco_df,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Calculate novelty for each item. Novelty is computed as the minus logarithm of
    (number of interactions with item / total number of interactions). The definition of the metric
    is based on the following reference using the choice model (eqs. 1 and 6):

    :Citation:

        P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
        choice, discovery and relevance, ECIR 2011

    The novelty of an item can be defined relative to a set of observed events on the set of all items.
    These can be events of user choice (item "is picked" by a random user) or user discovery
    (item "is known" to a random user). The above definition of novelty reflects a factor of item popularity.
    High novelty values correspond to long-tail items in the density function, that few users have interacted
    with and low novelty values correspond to popular head items.

    Returns:
        pandas.DataFrame: A dataframe with the following columns: col_item, item_novelty.
    """

    n_records = train_df.shape[0]
    item_count = pd.DataFrame(
        {"count": train_df.groupby([col_item]).size()}
    ).reset_index()
    item_count["item_novelty"] = -np.log2(item_count["count"] / n_records)
    df_item_novelty = (
        item_count[[col_item, "item_novelty"]]
        .sort_values(col_item)
        .reset_index(drop=True)
    )

    return df_item_novelty


@check_column_dtypes_novelty_coverage
def novelty(train_df, reco_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL):
    """Calculate the average novelty in a list of recommended items (this assumes that the recommendation list
    is already computed). Follows section 5 from

    :Citation:

        P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
        choice, discovery and relevance, ECIR 2011

    Returns:
        float: novelty.
    """

    df_item_novelty = historical_item_novelty(train_df, reco_df, col_user, col_item)
    n_recommendations = reco_df.shape[0]
    reco_item_count = pd.DataFrame(
        {"count": reco_df.groupby([col_item]).size()}
    ).reset_index()
    reco_item_novelty = reco_item_count.merge(df_item_novelty, on=col_item)
    reco_item_novelty["product"] = (
        reco_item_novelty["count"] * reco_item_novelty["item_novelty"]
    )
    avg_novelty = reco_item_novelty.agg({"product": "sum"})[0] / n_recommendations

    return avg_novelty


# Serendipity metrics
@check_column_dtypes_diversity_serendipity
@lru_cache_df(maxsize=1)
def user_item_serendipity(
    train_df,
    reco_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
    col_relevance=None,
):
    """Calculate serendipity of each item in the recommendations for each user.
    The metric definition is based on the following references:

    :Citation:

    Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
    introducing serendipity into music recommendation, WSDM 2012

    Eugene Yan, Serendipity: Accuracy’s unpopular best friend in Recommender Systems,
    eugeneyan.com, April 2020

    Returns:
        pandas.DataFrame: A dataframe with columns: col_user, col_item, user_item_serendipity.
    """
    # for every col_user, col_item in reco_df, join all interacted items from train_df.
    # These interacted items are repeated for each item in reco_df for a specific user.
    df_cosine_similarity = _get_cosine_similarity(
        train_df,
        item_feature_df,
        item_sim_measure,
        col_item_features,
        col_user,
        col_item,
        col_sim,
    )
    reco_user_item = reco_df[[col_user, col_item]]
    reco_user_item["reco_item_tmp"] = reco_user_item[col_item]

    train_user_item = train_df[[col_user, col_item]]
    train_user_item.columns = [col_user, "train_item_tmp"]

    reco_train_user_item = reco_user_item.merge(train_user_item, on=[col_user])
    reco_train_user_item["i1"] = reco_train_user_item[
        ["reco_item_tmp", "train_item_tmp"]
    ].min(axis=1)
    reco_train_user_item["i2"] = reco_train_user_item[
        ["reco_item_tmp", "train_item_tmp"]
    ].max(axis=1)

    reco_train_user_item_sim = reco_train_user_item.merge(
        df_cosine_similarity, on=["i1", "i2"], how="left"
    )
    reco_train_user_item_sim[col_sim].fillna(0, inplace=True)

    reco_user_item_avg_sim = (
        reco_train_user_item_sim.groupby([col_user, col_item])
        .agg({col_sim: "mean"})
        .reset_index()
    )
    reco_user_item_avg_sim.columns = [
        col_user,
        col_item,
        "avg_item2interactedHistory_sim",
    ]

    df_user_item_serendipity = reco_user_item_avg_sim.merge(
        reco_df, on=[col_user, col_item]
    )
    df_user_item_serendipity["user_item_serendipity"] = (
        1 - df_user_item_serendipity["avg_item2interactedHistory_sim"]
    ) * df_user_item_serendipity[col_relevance]
    df_user_item_serendipity = (
        df_user_item_serendipity[[col_user, col_item, "user_item_serendipity"]]
        .sort_values([col_user, col_item])
        .reset_index(drop=True)
    )

    return df_user_item_serendipity


@lru_cache_df(maxsize=1)
@check_column_dtypes_diversity_serendipity
def user_serendipity(
    train_df,
    reco_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
    col_relevance=None,
):
    """Calculate average serendipity for each user's recommendations.

    Returns:
        pandas.DataFrame: A dataframe with following columns: col_user, user_serendipity.
    """
    df_user_item_serendipity = user_item_serendipity(
        train_df,
        reco_df,
        item_feature_df,
        item_sim_measure,
        col_item_features,
        col_user,
        col_item,
        col_sim,
        col_relevance,
    )
    df_user_serendipity = (
        df_user_item_serendipity.groupby(col_user)
        .agg({"user_item_serendipity": "mean"})
        .reset_index()
    )
    df_user_serendipity.columns = [col_user, "user_serendipity"]
    df_user_serendipity = df_user_serendipity.sort_values(col_user).reset_index(
        drop=True
    )

    return df_user_serendipity


@check_column_dtypes_diversity_serendipity
def serendipity(
    train_df,
    reco_df,
    item_feature_df=None,
    item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
    col_item_features=DEFAULT_ITEM_FEATURES_COL,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_sim=DEFAULT_SIMILARITY_COL,
    col_relevance=None,
):
    """Calculate average serendipity for recommendations across all users.

    Returns:
        float: serendipity.
    """
    df_user_serendipity = user_serendipity(
        train_df,
        reco_df,
        item_feature_df,
        item_sim_measure,
        col_item_features,
        col_user,
        col_item,
        col_sim,
        col_relevance,
    )
    avg_serendipity = df_user_serendipity.agg({"user_serendipity": "mean"})[0]
    return avg_serendipity


# Coverage metrics
@check_column_dtypes_novelty_coverage
def catalog_coverage(
    train_df, reco_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL
):
    """Calculate catalog coverage for recommendations across all users.
    The metric definition is based on the "catalog coverage" definition in the following reference:

    :Citation:

        G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
        Recommender Systems Handbook pp. 257-297, 2010.

    Returns:
        float: catalog coverage
    """
    # distinct item count in reco_df
    count_distinct_item_reco = reco_df[col_item].nunique()
    # distinct item count in train_df
    count_distinct_item_train = train_df[col_item].nunique()

    # catalog coverage
    c_coverage = count_distinct_item_reco / count_distinct_item_train
    return c_coverage


@check_column_dtypes_novelty_coverage
def distributional_coverage(
    train_df, reco_df, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL
):
    """Calculate distributional coverage for recommendations across all users.
    The metric definition is based on formula (21) in the following reference:

    :Citation:

        G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
        Recommender Systems Handbook pp. 257-297, 2010.

    Returns:
        float: distributional coverage
    """
    # In reco_df, how  many times each col_item is being recommended
    df_itemcnt_reco = pd.DataFrame(
        {"count": reco_df.groupby([col_item]).size()}
    ).reset_index()

    # the number of total recommendations
    count_row_reco = reco_df.shape[0]

    df_entropy = df_itemcnt_reco
    df_entropy["p(i)"] = df_entropy["count"] / count_row_reco
    df_entropy["entropy(i)"] = df_entropy["p(i)"] * np.log2(df_entropy["p(i)"])

    d_coverage = -df_entropy.agg({"entropy(i)": "sum"})[0]

    return d_coverage
