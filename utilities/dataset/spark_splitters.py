"""
Collection of Splitter Methods
"""
import numpy as np

from pyspark.sql import Window
from pyspark.sql.functions import col, row_number, broadcast

from utilities.common.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_TIMESTAMP_COL,
)
from utilities.dataset.split_utils import (
    process_split_ratio,
    min_rating_filter,
)


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
    multi_split, ratio = process_split_ratio(ratio)

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

    multi_split, ratio = process_split_ratio(ratio)

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


