# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL
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


def df_to_libffm(df, col_rating=DEFAULT_RATING_COL, filepath=None):
    """Converts an input Dataframe (df) to another Dataframe (df) in libffm format. A text file of the converted
    Dataframe is optionally generated.

    Note:
        The input dataframe is expected to represent the feature data in the following schema
        |field-1|field-2|...|field-n|rating|
        |feature-1-1|feature-2-1|...|feature-n-1|1|
        |feature-1-2|feature-2-2|...|feature-n-2|0|
        ...
        |feature-1-i|feature-2-j|...|feature-n-k|0|
        Where
        1. each "field-*" occupies one column in the data, and
        2. "feature-*-*" can be either a string or a numerical value.

        The above data will be converted to the libffm format by following the convention as explained in
        https://www.csie.ntu.edu.tw/~r01922136/slides/ffm.pdf

    Args:
        df (pd.DataFrame): input Pandas dataframe.
        col_rating (str): rating of the data.
        filepath (str): path to save the converted data.

    Return:
        pd.DataFrame: data in libffm format.
    """
    df_new = df.copy()

    # Encode field (rating column is not included)
    field_names = list(df_new.drop(col_rating, axis=1).columns)
    field_dict = {k: v+1 for v, k in enumerate(field_names)}

    # Encode field-feature
    field_feature_list = []
    for col in field_names:
        if df_new[col].dtype == int:
            df_new[col] = df_new[col].astype(str)
        if df_new[col].dtype == object:
            field_feature_concat = ['{}-{}'.format(col, x) for x in df_new[col].values]
            field_feature_list.extend(field_feature_concat)
    field_feature_dict = {k: v+1 for v, k in enumerate(field_feature_list)}

    def _convert(field, feature, field_index_dict, field_feature_index_dict):
        field_index = field_index_dict[field]
        field_feature_index = (
            field_feature_index_dict['{}-{}'.format(field, feature)]
            if isinstance(feature, str)
            else None
        )

        return (
            "{}:{}:1".format(field_index, field_feature_index)
            if isinstance(feature, str)
            else "{}:{}:{}".format(field_index, field_index, feature)
        )

    for col in field_names:
        if col is not col_rating:
            df_new[col] = df_new[col].apply(lambda x: _convert(col, x, field_dict, field_feature_dict))

    # Move rating column to the first.
    field_names.append(col_rating)
    field_names.insert(0, field_names.pop(field_names.index(col_rating)))
    df_new = df_new[field_names]

    if filepath is not None:
        np.savetxt(filepath, df_new.values, delimiter=' ', fmt='%s')

    return df_new


