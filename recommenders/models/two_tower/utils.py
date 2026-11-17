# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging

import numpy as np
import pandas as pd
import torch

from recommenders.utils.constants import (
    DEFAULT_GENRE_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
)

logger = logging.getLogger(__name__)


def recommend_k_items(
    model,
    data,
    users=None,
    top_k=10,
    remove_seen=True,
    batch_size=1024,
    col_prediction=DEFAULT_PREDICTION_COL,
):
    """Retrieve the top-k items for each user.

    All item vectors are computed once and every user vector is scored against
    them with a matrix product, followed by `torch.topk`. For very large
    catalogues, the vectors from `model.encode_items()` can instead be loaded
    into an approximate nearest-neighbour index.

    Args:
        model (TwoTowerModel): Trained model.
        data (TwoTowerData): The data object the model was trained with. It
            provides the ID mappings and the items each user has already seen.
        users (list-like): Raw IDs of the users to recommend for. If None, all
            training users are used. Users unknown to the model are skipped.
        top_k (int): Number of items to return per user.
        remove_seen (bool): Whether to exclude items the user interacted with
            in the training data.
        batch_size (int): Number of users scored at a time.
        col_prediction (str): Name of the score column of the output.

    Returns:
        pandas.DataFrame: Columns user, item and score, with up to `top_k` rows
        per user, sorted by user and by descending score.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    if users is None:
        user_idx = np.arange(data.n_users)
    else:
        users = pd.unique(pd.Series(users))
        mapped = pd.Series(users).map(data.user2idx)
        n_unknown = int(mapped.isna().sum())
        if n_unknown > 0:
            logger.warning(f"Skipping {n_unknown} users not seen during training")
        user_idx = mapped.dropna().to_numpy(dtype=np.int64)

    columns = [data.col_user, data.col_item, col_prediction]
    if len(user_idx) == 0:
        return pd.DataFrame(columns=columns)

    k = min(top_k, data.n_items)
    device = next(model.parameters()).device
    model.eval()

    out_users, out_items, out_scores = [], [], []
    with torch.no_grad():
        item_vectors = model.encode_items()
        for start in range(0, len(user_idx), batch_size):
            batch = user_idx[start : start + batch_size]
            user_vectors = model.encode_users(
                torch.tensor(batch, dtype=torch.long, device=device)
            )
            scores = user_vectors @ item_vectors.T
            if remove_seen:
                seen = data.seen[batch].tocoo()
                rows = torch.as_tensor(seen.row, dtype=torch.long, device=device)
                cols = torch.as_tensor(seen.col, dtype=torch.long, device=device)
                scores[rows, cols] = -float("inf")
            top_scores, top_items = torch.topk(scores, k, dim=1)
            top_scores = top_scores.cpu().numpy()
            top_items = top_items.cpu().numpy()
            # Users who have seen almost every item get fewer than k results
            valid = np.isfinite(top_scores)
            out_users.append(np.repeat(batch, k).reshape(-1, k)[valid])
            out_items.append(top_items[valid])
            out_scores.append(top_scores[valid])

    return pd.DataFrame(
        {
            data.col_user: data.users[np.concatenate(out_users)],
            data.col_item: data.items[np.concatenate(out_items)],
            col_prediction: np.concatenate(out_scores),
        }
    )


def multi_hot_features(
    df, col_id=DEFAULT_ITEM_COL, col_values=DEFAULT_GENRE_COL, sep="|", prefix=""
):
    """Turn a column of delimited labels (e.g. MovieLens genres) into multi-hot features.

    Args:
        df (pandas.DataFrame): DataFrame with an ID column and a label column,
            e.g. "Action|Comedy".
        col_id (str): ID column name.
        col_values (str): Name of the column holding the delimited labels.
        sep (str): Label separator.
        prefix (str): Prefix added to the generated feature column names.

    Returns:
        pandas.DataFrame: One row per ID, with the ID column and one 0/1 column
        per label. It can be passed as `user_features` or `item_features` to
        `TwoTowerData`.
    """
    unique = df[[col_id, col_values]].drop_duplicates(subset=col_id)
    dummies = unique[col_values].fillna("").astype(str).str.get_dummies(sep=sep)
    dummies = dummies.add_prefix(prefix).astype(np.float32)
    return pd.concat(
        [unique[[col_id]].reset_index(drop=True), dummies.reset_index(drop=True)],
        axis=1,
    )
