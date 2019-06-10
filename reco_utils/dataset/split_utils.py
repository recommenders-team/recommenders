# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from reco_utils.common.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL

try:
    from pyspark.sql.functions import col, broadcast
except ImportError:
    pass  # so the environment without spark doesn't break


def process_split_ratio(ratio):
    """Generate split ratio lists

    Args:
        ratio (float or list): a float number that indicates split ratio or a list of float
        numbers that indicate split ratios (if it is a multi-split).

    Returns:
        tuple: a tuple containing
            bool: A boolean variable multi that indicates if the splitting is multi or single.
            list: A list of normalized split ratios.
    """
    if isinstance(ratio, float):
        if ratio <= 0 or ratio >= 1:
            raise ValueError("Split ratio has to be between 0 and 1")

        multi = False
    elif isinstance(ratio, list):
        if any([x <= 0 for x in ratio]):
            raise ValueError(
                "All split ratios in the ratio list should be larger than 0."
            )

        # normalize split ratios if they are not summed to 1
        if sum(ratio) != 1.0:
            ratio = [x / sum(ratio) for x in ratio]

        multi = True
    else:
        raise TypeError("Split ratio should be either float or a list of floats.")

    return multi, ratio


def min_rating_filter_pandas(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (pd.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating, 
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        pd.DataFrame: DataFrame with at least columns of user and item that has been 
            filtered by the given specifications.
    """
    split_by_column, _ = _check_min_rating_filter(
        filter_by, min_rating, col_user, col_item
    )
    rating_filtered = data.groupby(split_by_column).filter(
        lambda x: len(x) >= min_rating
    )
    return rating_filtered


def min_rating_filter_spark(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.
    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is called warm if he has rated at least 4 items.

    Args:
        data (spark.DataFrame): DataFrame of user-item tuples. Columns of user and item
            should be present in the DataFrame while other columns like rating, 
            timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        spark.DataFrame: DataFrame with at least columns of user and item that has been 
            filtered by the given specifications.
    """
    split_by_column, split_with_column = _check_min_rating_filter(
        filter_by, min_rating, col_user, col_item
    )
    rating_temp = (
        data.groupBy(split_by_column)
        .agg({split_with_column: "count"})
        .withColumnRenamed("count(" + split_with_column + ")", "n" + split_with_column)
        .where(col("n" + split_with_column) >= min_rating)
    )

    rating_filtered = data.join(broadcast(rating_temp), split_by_column).drop(
        "n" + split_with_column
    )
    return rating_filtered


def _check_min_rating_filter(filter_by, min_rating, col_user, col_item):
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    split_by_column = col_user if filter_by == "user" else col_item
    split_with_column = col_item if filter_by == "user" else col_user
    return split_by_column, split_with_column


def split_pandas_data_with_ratios(data, ratios, seed=42, shuffle=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from
        https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test

    Args:
        data (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split. The ratios have to sum to 1.
        seed (int): random seed.
        shuffle (bool): whether data will be shuffled when being split.

    Returns:
        list: List of pd.DataFrame split by the given specifications.
    """
    if sum(ratios) != 1.0:
        raise ValueError("The ratios have to sum to 1")

    split_index = np.cumsum(ratios).tolist()[:-1]

    if shuffle:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    # Add split index (this makes splitting by group more efficient).
    for i in range(len(ratios)):
        splits[i]["split_index"] = i

    return splits
