# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
from cornac.models import BPR as CBPR

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)


class BPR(CBPR):
    """Custom BPR class extending Cornac's BPR model with a recommend_k_items method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def recommend_k_items(
        self,
        data,
        top_k=10,
        remove_seen=False,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
    ):
        """Computes predictions of recommender model from Cornac on all users and items in data.
            It can be used for computing ranking metrics like NDCG.

        Args:
            data (pandas.DataFrame): The data from which to get the users and items
            usercol (str): Name of the user column
            itemcol (str): Name of the item column
            predcol (str): Name of the prediction column
            remove_seen (bool): Flag to remove (user, item) pairs seen in the training data

        Returns:
            pandas.DataFrame: Dataframe with usercol, itemcol, predcol
        """
        # Precompute items and users
        items = list(self.train_set.iid_map.keys())
        users = list(self.train_set.uid_map.keys())
        n_users = len(users)
        n_items = len(items)

        # Compute full score matrix in one go
        user_indices = [self.train_set.uid_map[u] for u in users]
        item_indices = [self.train_set.iid_map[i] for i in items]
        U = self.u_factors[user_indices]
        V = self.i_factors[item_indices]
        B = self.i_biases[item_indices]

        # Matrix multiplication for all user-item pairs
        preds_matrix = U @ V.T + B
        user_array = np.repeat(users, n_items)
        item_array = np.tile(items, n_users)
        preds = preds_matrix.flatten()

        all_predictions = pd.DataFrame(
            {col_user: user_array, col_item: item_array, col_rating: preds}
        )

        if remove_seen:
            seen = data[[col_user, col_item]].drop_duplicates()
            merged = all_predictions.merge(
                seen, on=[col_user, col_item], how="left", indicator=True
            )
            return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
        return all_predictions
