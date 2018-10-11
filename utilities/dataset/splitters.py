"""
Collection of Splitter Methods
"""
import numpy as np
import pandas as pd

# pylint: disable=E0611

import pyspark
from pyspark.sql import Window
from pyspark.sql.functions import col, row_number, broadcast
from pyspark.sql.functions import round as spark_round
from sklearn.model_selection import train_test_split as sk_split
import surprise
from surprise.model_selection import train_test_split

from utilities.common.constants import (
    DEFAULT_RATING_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_TIMESTAMP_COL,
)


def _process_split_ratio(ratio):
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


def spark_random_split(data, ratio=0.75, seed=123):
    """Spark random splitter

    Args:
        data (spark.DataFrame): Spark DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
        it splits data into
        two halfs and the ratio argument indicates the ratio of training data set;
        if it is a list of float numbers, the splitter splits data into several portions
        corresponding to the
        split ratios. If a list is provided and the ratios are not summed to 1, they will be
        normalized.
        seed (int): Seed.

    Returns:
        Splits of the input data.
    """
    multi_split, ratio = _process_split_ratio(ratio)

    if multi_split:
        return data.randomSplit(ratio, seed=seed)
    else:
        return data.randomSplit([ratio, 1 - ratio], seed=seed)


def spark_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    """Spark chronological splitter

    Args:
        data (spark.DataFrame): Spark DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
        it splits data into
        two halfs and the ratio argument indicates the ratio of training data set;
        if it is a list of float numbers, the splitter splits data into several portions
        corresponding to the
        split ratios. If a list is provided and the ratios are not summed to 1, they will be
        normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to filter
        with
        min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
        col_timestamp (str): column name of timestamps.

    Returns:
        Splits of the input data.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    multi_split, ratio = _process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

    if min_rating > 1:
        data = min_rating_filter(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )

    ratio = ratio if multi_split else [ratio, 1 - ratio]
    ratio_index = np.cumsum(ratio)

    window_spec = Window.partitionBy(split_by_column).orderBy(col(col_timestamp).desc())

    rating_grouped = (
        data.groupBy(split_by_column)
        .agg({col_timestamp: "count"})
        .withColumnRenamed("count(" + col_timestamp + ")", "count")
    )
    rating_all = data.join(broadcast(rating_grouped), on=split_by_column)

    rating_rank = rating_all.withColumn(
        "rank", row_number().over(window_spec) / col("count")
    )

    splits = []
    for i, _ in enumerate(ratio_index):
        if i == 0:
            rating_split = rating_rank.filter(col("rank") <= ratio_index[i])
        else:
            rating_split = rating_rank.filter(
                (col("rank") <= ratio_index[i]) & (col("rank") > ratio_index[i - 1])
            )

        splits.append(rating_split)

    return splits


def pandas_random_split(data, ratio=0.75, seed=123):
    """Pandas random splitter

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
        it splits data into
        two halfs and the ratio argument indicates the ratio of training data set;
        if it is a list of float numbers, the splitter splits data into several portions
        corresponding to the
        split ratios. If a list is provided and the ratios are not summed to 1, they will be
        normalized.
        seed (int): Seed.

    Returns:
        Splits of the input data.
    """
    multi_split, ratio = _process_split_ratio(ratio)

    if multi_split:
        splits = _split_pandas_data_with_ratios(data, ratio, resample=True, seed=seed)
        return splits
    else:
        return sk_split(data, test_size=None, train_size=ratio, random_state=seed)


def pandas_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    """Pandas chronological splitter

    Args:
        data (pandas.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
        it splits data into
        two halfs and the ratio argument indicates the ratio of training data set;
        if it is a list of float numbers, the splitter splits data into several portions
        corresponding to the
        split ratios. If a list is provided and the ratios are not summed to 1, they will be
        normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to filter
        with
        min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
        col_timestamp (str): column name of timestamps.

    Returns:
        Splits of the input data.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    multi_split, ratio = _process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

    # Sort data by timestamp.
    data = data.sort_values(
        by=[split_by_column, col_timestamp], axis=0, ascending=False
    )

    ratio = ratio if multi_split else [ratio, 1 - ratio]

    if min_rating > 1:
        data = min_rating_filter(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )

    num_of_splits = len(ratio)
    splits = [pd.DataFrame({})] * num_of_splits
    df_grouped = data.sort_values(col_timestamp).groupby(split_by_column)
    for name, group in df_grouped:
        group_splits = _split_pandas_data_with_ratios(
            df_grouped.get_group(name), ratio, resample=False
        )
        for x in range(num_of_splits):
            splits[x] = pd.concat([splits[x], group_splits[x]])

    return splits


def min_rating_filter(
    data,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
):
    """Filter rating DataFrame for each user with minimum rating.

    Filter rating data frame with minimum number of ratings for user/item is usually useful to
    generate a
    new data frame with warm user/item. The warmth is defined by min_rating argument. For
    example, a user is
    called warm if he has rated at least 4 items.

    Args:
        data (spark.DataFrame or pandas.DataFrame): Spark data frame or Pandas data frame of
        user-item tuples. Columns of user and item should be
        present in the data frame while other columns like rating, timestamp, etc. can be optional.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to filter with
        min_rating.
        col_user (str): column name of user ID.
        col_item (str): column name of item ID.

    Returns:
        Spark DataFrame with at least columns of user and item that has been filtered by the
        given specifications.
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


def _split_pandas_data_with_ratios(data, ratios, seed=1234, resample=False):
    """Helper function to split pandas DataFrame with given ratios

    Note:
        Implementation referenced from
        https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train
        -validation-and-test

    Args:
        data (pandas.DataFrame): Pandas data frame to be split.
        ratios (list of floats): list of ratios for split.
        seed (int): random seed.
        resample (bool): whether data will be resampled when being split.

    Returns:
        List of data frames which are split by the given specifications.
    """
    split_index = np.cumsum(ratios).tolist()[:-1]

    if resample:
        data = data.sample(frac=1, random_state=seed)

    splits = np.split(data, [round(x * len(data)) for x in split_index])

    return splits
