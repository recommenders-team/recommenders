# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from sklearn.model_selection import train_test_split as sk_split

from reco_utils.common.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_USER_COL,
    DEFAULT_TIMESTAMP_COL,
)
from reco_utils.dataset.split_utils import (
    process_split_ratio,
    min_rating_filter,
    split_pandas_data_with_ratios,
)


def python_random_split(data, ratio=0.75, seed=123):
    """Pandas random splitter
    The splitter randomly splits the input data.

    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halfs and the ratio argument indicates the ratio 
            of training data set; if it is a list of float numbers, the splitter splits 
            data into several portions corresponding to the split ratios. If a list is 
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        
    Returns:
        list: Splits of the input data as pd.DataFrame.
    """
    multi_split, ratio = process_split_ratio(ratio)

    if multi_split:
        return split_pandas_data_with_ratios(data, ratio, resample=True, seed=seed)
    else:
        return sk_split(data, test_size=None, train_size=ratio, random_state=seed)


def python_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    """Pandas chronological splitter
    This function splits data in a chronological manner. That is, for each user / item, the
    split function takes proportions of ratings which is specified by the split ratio(s).
    The split is stratified.

    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halfs and the ratio argument indicates the ratio of 
            training data set; if it is a list of float numbers, the splitter splits 
            data into several portions corresponding to the split ratios. If a list is 
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to 
            filter with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.
        col_timestamp (str): column name of timestamps.

    Returns:
        list: Splits of the input data as pd.DataFrame.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    multi_split, ratio = process_split_ratio(ratio)

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
        group_splits = split_pandas_data_with_ratios(
            df_grouped.get_group(name), ratio, resample=False
        )
        for x in range(num_of_splits):
            splits[x] = pd.concat([splits[x], group_splits[x]])

    return splits


def python_stratified_split(
        data,
        ratio=0.75,
        min_rating=1,
        filter_by="user",
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL
):
    """Pandas stratified splitter
    For each user / item, the split function takes proportions of ratings which is
    specified by the split ratio(s). The split is stratified.

    Args:
        data (pd.DataFrame): Pandas DataFrame to be split.
        ratio (float or list): Ratio for splitting data. If it is a single float number
            it splits data into two halfs and the ratio argument indicates the ratio of
            training data set; if it is a list of float numbers, the splitter splits
            data into several portions corresponding to the split ratios. If a list is
            provided and the ratios are not summed to 1, they will be normalized.
        seed (int): Seed.
        min_rating (int): minimum number of ratings for user or item.
        filter_by (str): either "user" or "item", depending on which of the two is to
            filter with min_rating.
        col_user (str): column name of user IDs.
        col_item (str): column name of item IDs.

    Returns:
        list: Splits of the input data as pd.DataFrame.
    """
    if not (filter_by == "user" or filter_by == "item"):
        raise ValueError("filter_by should be either 'user' or 'item'.")

    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")

    multi_split, ratio = process_split_ratio(ratio)

    split_by_column = col_user if filter_by == "user" else col_item

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
    df_grouped = data.groupby(split_by_column)
    for name, group in df_grouped:
        group_splits = split_pandas_data_with_ratios(
            df_grouped.get_group(name), ratio, resample=True
        )
        for x in range(num_of_splits):
            splits[x] = pd.concat([splits[x], group_splits[x]])

    return splits
