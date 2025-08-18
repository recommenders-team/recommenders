# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
import torch

from recommenders.utils import constants as cc


def cartesian_product(*arrays):
    """Compute the Cartesian product. This is a helper function.

    Args:
        arrays (tuple of numpy.ndarray): Input arrays

    Returns:
        numpy.ndarray: product

    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def score(
    model,
    test_df,
    user_col=cc.DEFAULT_USER_COL,
    item_col=cc.DEFAULT_ITEM_COL,
    prediction_col=cc.DEFAULT_PREDICTION_COL,
    top_k=None,
):
    """Score all users+items provided and reduce to top_k items per user if top_k>0

    Args:
        model (object): Model.
        test_df (pandas.DataFrame): Test dataframe.
        user_col (str): User column name.
        item_col (str): Item column name.
        prediction_col (str): Prediction column name.
        top_k (int): Number of top items to recommend.

    Returns:
        pandas.DataFrame: Result of recommendation
    """
    # replace values not known to the model with NaN
    total_users, total_items = model.classes.values()
    test_df.loc[~test_df[user_col].isin(total_users), user_col] = np.nan
    test_df.loc[~test_df[item_col].isin(total_items), item_col] = np.nan

    # map ids to embedding ids
    u = model._get_idx(test_df[user_col], is_item=False)
    m = model._get_idx(test_df[item_col], is_item=True)

    # score the pytorch model
    x = torch.column_stack((u, m))

    if torch.cuda.is_available():
        x = x.to("cuda")
        model = model.to("cuda")

    pred = model.forward(x).detach().cpu().numpy()
    scores = pd.DataFrame(
        {user_col: test_df[user_col], item_col: test_df[item_col], prediction_col: pred}
    )
    scores = scores.sort_values([user_col, prediction_col], ascending=[True, False])

    if top_k is not None:
        top_scores = scores.groupby(user_col).head(top_k).reset_index(drop=True)
    else:
        top_scores = scores

    return top_scores
