# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import numpy as np
import pandas as pd
import fastai
from fastprogress import force_console_behavior
import fastprogress


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


def score(learner, test_df, user_col, item_col, prediction_col, top_k=0):
    """score all users+movies provided and reduce to top_k items per user if top_k>0"""
    # replace values not known to the model with #na#
    total_users, total_items = learner.data.train_ds.x.classes.values()
    test_df.loc[~test_df[user_col].isin(total_users), user_col] = total_users[0]
    test_df.loc[~test_df[item_col].isin(total_items), item_col] = total_items[0]

    # map ids to embedding ids
    u = learner.get_idx(test_df[user_col], is_item=False)
    m = learner.get_idx(test_df[item_col], is_item=True)

    # score the pytorch model
    pred = learner.model.forward(u, m)
    scores = pd.DataFrame(
        {user_col: test_df[user_col], item_col: test_df[item_col], prediction_col: pred}
    )
    scores = scores.sort_values([user_col, prediction_col], ascending=[True, False])
    if top_k > 0:
        top_scores = scores.groupby(user_col).head(top_k).reset_index(drop=True)
    else:
        top_scores = scores
    return top_scores


def hide_fastai_progress_bar():
    """Hide fastai progress bar"""
    fastprogress.fastprogress.NO_BAR = True
    fastprogress.fastprogress.WRITER_FN = str
    master_bar, progress_bar = force_console_behavior()
    fastai.basic_train.master_bar, fastai.basic_train.progress_bar = (
        master_bar,
        progress_bar,
    )

