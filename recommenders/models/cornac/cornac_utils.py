# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import numpy as np

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
)


def predict(
    model,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
):
    """Computes predictions of a recommender model from Cornac on the data.
    Can be used for computing rating metrics like RMSE.

    Args:
        model (cornac.models.Recommender): A recommender model from Cornac
        data (pandas.DataFrame): The data on which to predict
        usercol (str): Name of the user column
        itemcol (str): Name of the item column

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    uid_map = model.train_set.uid_map
    iid_map = model.train_set.iid_map
    predictions = [
        [
            getattr(row, usercol),
            getattr(row, itemcol),
            model.rate(
                user_idx=uid_map.get(getattr(row, usercol), len(uid_map)),
                item_idx=iid_map.get(getattr(row, itemcol), len(iid_map)),
            ),
        ]
        for row in data.itertuples()
    ]
    predictions = pd.DataFrame(data=predictions, columns=[usercol, itemcol, predcol])
    return predictions


def predict_ranking(
    model,
    data,
    usercol=DEFAULT_USER_COL,
    itemcol=DEFAULT_ITEM_COL,
    predcol=DEFAULT_PREDICTION_COL,
    remove_seen=False,
):
    """Computes predictions of recommender model from Cornac on all users and items in data.
    It can be used for computing ranking metrics like NDCG.

    Args:
        model (cornac.models.Recommender): A recommender model from Cornac
        data (pandas.DataFrame): The data from which to get the users and items
        usercol (str): Name of the user column
        itemcol (str): Name of the item column
        predcol (str): Name of the prediction column
        remove_seen (bool): Flag to remove (user, item) pairs seen in the training data

    Returns:
        pandas.DataFrame: Dataframe with usercol, itemcol, predcol
    """
    # Precompute items and users
    items = list(model.train_set.iid_map.keys())
    users = list(model.train_set.uid_map.keys())
    n_users = len(users)
    n_items = len(items)

    # Compute full score matrix in one go
    user_indices = [model.train_set.uid_map[u] for u in users]
    item_indices = [model.train_set.iid_map[i] for i in items]
    U = model.u_factors[user_indices]
    V = model.i_factors[item_indices]
    B = model.i_biases[item_indices]

    # Matrix multiplication for all user-item pairs
    preds_matrix = U @ V.T + B
    user_array = np.repeat(users, n_items)
    item_array = np.tile(items, n_users)
    preds = preds_matrix.flatten()

    all_predictions = pd.DataFrame({
        usercol: user_array,
        itemcol: item_array,
        predcol: preds
    })

    if remove_seen:
        seen = data[[usercol, itemcol]].drop_duplicates()
        merged = all_predictions.merge(seen, on=[usercol, itemcol], how='left', indicator=True)
        return merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
    return all_predictions