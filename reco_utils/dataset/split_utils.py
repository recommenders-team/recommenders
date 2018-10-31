# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from reco_utils.common.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL


def process_split_ratio(ratio):
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


def min_rating_filter(
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
        data (spark.DataFrame or pandas.DataFrame): Spark DataFrame or Pandas DataFrame of
            user-item tuples. Columns of user and item should be present in the data 
            frame while other columns like rating, timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        spark.DataFrame: DataFrame with at least columns of user and item that has been 
            filtered by the given specifications.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    split_by_column = col_user if filter_by == "user" else col_item
    split_with_column = col_item if filter_by == "user" else col_user

    if isinstance(data, pd.DataFrame):
        rating_filtered = data.groupby(split_by_column).filter(
            lambda x: len(x) >= min_rating
        )

        return rating_filtered
    try:
        import pyspark
        from pyspark.sql.functions import col, broadcast

        if isinstance(data, pyspark.sql.DataFrame):
            rating_temp = (
                data.groupBy(split_by_column)
                .agg({split_with_column: "count"})
                .withColumnRenamed(
                    "count(" + split_with_column + ")", "n" + split_with_column
                )
                .where(col("n" + split_with_column) >= min_rating)
            )

            rating_filtered = data.join(broadcast(rating_temp), split_by_column).drop(
                "n" + split_with_column
            )

            return rating_filtered
    except NameError:
        raise TypeError("Spark not installed")
    raise TypeError(
        "Only Spark and Pandas Data Frames are supported for min rating filter."
    )


def split_pandas_data_with_ratios(data, ratios, seed=1234, resample=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from
        https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train
        -validation-and-test

    Args:
        data (pd.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split.
        seed (int): random seed.
        resample (bool): whether data will be resampled when being split.

    Returns:
        list: List of pd.DataFrame splitted by the given specifications.
    """
    split_index = np.cumsum(ratios).tolist()[:-1]

    if resample:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    return splits
