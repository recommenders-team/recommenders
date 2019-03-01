# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL
)


def user_item_pairs(
    user_df,
    item_df,
    user_col=DEFAULT_USER_COL,
    item_col=DEFAULT_ITEM_COL,
    user_item_filter_df=None,
    shuffle=True
):
    """Get all pairs of users and items data.

    Args:
        user_df (pd.DataFrame): User data containing unique user ids and maybe their features.
        item_df (pd.DataFrame): Item data containing unique item ids and maybe their features.
        user_col (str): User id column name.
        item_col (str): Item id column name.
        user_item_filter_df (pd.DataFrame): User-item pairs to be used as a filter.
        shuffle (bool): If True, shuffles the result.

    Returns:
        pd.DataFrame: All pairs of user-item from user_df and item_df, excepting the pairs in user_item_filter_df
    """

    # Get all user-item pairs
    user_df['key'] = 1
    item_df['key'] = 1
    users_items = user_df.merge(item_df, on='key')

    user_df.drop('key', axis=1, inplace=True)
    item_df.drop('key', axis=1, inplace=True)
    users_items.drop('key', axis=1, inplace=True)

    # Filter
    if user_item_filter_df is not None:
        user_item_col = [user_col, item_col]
        users_items = users_items.loc[
            ~users_items.set_index(user_item_col).index.isin(user_item_filter_df.set_index(user_item_col).index)
        ]

    if shuffle:
        users_items = users_items.sample(frac=1).reset_index(drop=True)

    return users_items


def filter_by(df, filter_by_df, filter_by_cols):
    """From the input DataFrame (df), remove the records whose target column (filter_by_cols) values are
    exist in the filter-by DataFrame (filter_by_df)

    Args:
        df (pd.DataFrame): Source dataframe.
        filter_by_df (pd.DataFrame): Filter dataframe.
        filter_by_cols (iterable of str): Filter columns.

    Returns:
        pd.DataFrame: Dataframe filtered by filter_by_df on filter_by_cols
    """

    return df.loc[
        ~df.set_index(filter_by_cols).index.isin(filter_by_df.set_index(filter_by_cols).index)
    ]
