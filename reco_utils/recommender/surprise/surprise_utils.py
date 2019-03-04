# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
from reco_utils.common.general_utils import invert_dictionary


def surprise_trainset_to_df(
    trainset, col_user="uid", col_item="iid", col_rating="rating"
):
    """Converts a surprise.Trainset object to pd.DataFrame
    
    Args:
        trainset (obj): A surprise.Trainset object.
        col_user (str): User column name.
        col_item (str): Item column name.
        col_rating (str): Rating column name.
    
    Returns:
        pd.DataFrame: A dataframe
    """
    df = pd.DataFrame(trainset.all_ratings(), columns=[col_user, col_item, col_rating])
    map_user = (
        trainset._inner2raw_id_users
        if trainset._inner2raw_id_users is not None
        else invert_dictionary(trainset._raw2inner_id_users)
    )
    map_item = (
        trainset._inner2raw_id_items
        if trainset._inner2raw_id_items is not None
        else invert_dictionary(trainset._raw2inner_id_items)
    )
    df[col_user] = df[col_user].map(map_user)
    df[col_item] = df[col_item].map(map_item)
    return df

