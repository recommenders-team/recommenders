# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pandas as pd

from reco_utils.common.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL
from reco_utils.common.general_utils import invert_dictionary


def surprise_trainset_to_df(
    trainset, col_user="uid", col_item="iid", col_rating="rating"
):
    """Converts a surprise.Trainset object to pd.DataFrame
    More info: https://surprise.readthedocs.io/en/stable/trainset.html

    Args:
        trainset (obj): A surprise.Trainset object.
        col_user (str): User column name.
        col_item (str): Item column name.
        col_rating (str): Rating column name.
    
    Returns:
        pd.DataFrame: A dataframe. The user and item columns are strings and the rating columns are floats.
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


def compute_rating_predictions(algo, data, usercol=DEFAULT_USER_COL, itemcol=DEFAULT_ITEM_COL, predcol=DEFAULT_PREDICTION_COL):
    """Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.
    
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column
    
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    predictions = [algo.predict(getattr(row, usercol), getattr(row, itemcol)) for row in data.itertuples()]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(index=str, columns={'uid': usercol, 'iid': itemcol, 'est': predcol})
    return predictions.drop(['details', 'r_ui'], axis='columns')


def compute_ranking_predictions(algo, 
                                data, 
                                usercol=DEFAULT_USER_COL, 
                                itemcol=DEFAULT_ITEM_COL,
                                predcol=DEFAULT_PREDICTION_COL, 
                                remove_seen=False):
    """Computes predictions of an algorithm from Surprise on all users and items in data. can be used for computing
    ranking metrics like NDCG.
    
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        remove_seen (bool): flag to remove (user, item) pairs seen in the training data
    
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    preds_lst = []
    for user in data[usercol].unique():
        for item in data[itemcol].unique():
            preds_lst.append([user, item, algo.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if remove_seen:
        tempdf = pd.concat([data[[usercol, itemcol]],
                            pd.DataFrame(data=np.ones(data.shape[0]), columns=['dummycol'], index=data.index)],
                            axis=1)
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged['dummycol'].isnull()].drop('dummycol', axis=1)
    else:
        return all_predictions
