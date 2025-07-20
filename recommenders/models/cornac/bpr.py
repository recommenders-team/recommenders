# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from cornac.models import BPR as CBPR

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL
)

class BPR(CBPR):
    """Custom BPR class extending Cornac's BPR model with a recommend_k_items method."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def recommend_k_items(
        self,
        data,
        top_k=None,
        remove_seen=False,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
    ):
        """Computes top-k predictions of recommender model from Cornac on all users in data.
        It can be used for computing ranking metrics like NDCG.

        Args:
            data (pandas.DataFrame): The data from which to get the users and items.
            top_k (int): Number of items to recommend per user.
            remove_seen (bool): Flag to remove (user, item) pairs seen in the training data.
            col_user (str): Name of the user column.
            col_item (str): Name of the item column.
            col_rating (str): Name of the prediction column.

        Returns:
            pandas.DataFrame: Dataframe with col_user, col_item, col_rating for top-k items per user.
        """
        # Get user and item mappings
        items = np.array(list(self.train_set.iid_map.keys()))
        users = np.array(list(self.train_set.uid_map.keys()))
        n_users = len(users)
        n_items = len(items)
        top_k = n_items if top_k is None else min(top_k, n_items) 

        # Compute user and item indices
        user_indices = np.array([self.train_set.uid_map[u] for u in users])
        item_indices = np.array([self.train_set.iid_map[i] for i in items])

        # Get latent factors and biases
        U = self.u_factors[user_indices]
        V = self.i_factors[item_indices]
        B = self.i_biases[item_indices] if hasattr(self, "i_biases") else np.zeros(n_items)

        # Compute score matrix for all user-item pairs
        preds_matrix = U @ V.T + B  # Shape: (n_users, n_items)

        # Select top-k items per user
        top_k_indices = np.argpartition(preds_matrix, -top_k, axis=1)[:, -top_k:]  # Shape: (n_users, top_k)
        sorted_indices = np.argsort(-preds_matrix[np.arange(n_users)[:, None], top_k_indices], axis=1)
        top_k_indices = top_k_indices[np.arange(n_users)[:, None], sorted_indices]  # Shape: (n_users, top_k)

        # Extract items and scores
        user_array = np.repeat(users, top_k)  # Shape: (n_users * top_k,)
        item_array = items[top_k_indices].flatten()  # Shape: (n_users * top_k,)
        pred_array = np.take_along_axis(preds_matrix, top_k_indices, axis=1).flatten()  # Shape: (n_users * top_k,)

        # Create DataFrame
        all_predictions = pd.DataFrame(
            {col_user: user_array, col_item: item_array, col_prediction: pred_array}
        )

        if remove_seen:
            seen = data[[col_user, col_item]].drop_duplicates()
            merged = all_predictions.merge(
                seen, on=[col_user, col_item], how="left", indicator=True
            )
            return merged[merged["_merge"] == "left_only"].drop(columns=["_merge"]).reset_index(drop=True)
        return all_predictions