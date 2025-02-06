# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.
#
# Based on https://github.com/microsoft/UniRec/
#

import numpy as np
import pandas as pd
from typing import List
from .morec_data_sampler import MoRecDS
from .morec_objective_controller import *


def load_morec_meta_data(n_items: int, filepath: str, objectives: List[str]):
    """
    Load the item meta file for MoRec.

    The file should be csv-format, which is expected to consist of all items and their meta information.
    MoRec now supports three kinds of objectives: ['revenue', 'fairness', 'alignment'], and the corresponding
    columns should be ["weight", "fair_group", "align_group"] respectively.

    In term of the item with index 0 (padding item), the weight would be set as 0.0, and fair_group, align_group
    would be set as 0. If the minimum value of *_group in the file is 0 for non-padding items, the group numbers would
    be increased by 1, as to leave the index 0 for padding item.

    Args:
        n_items (int): number of items (include padding item), used for checking whether there are items missing in the file.
                       Only padding item (index 0) is allowed to be missed in the file.
        filepath (str): the path of the item meta file.
        objectives (List[str]): objectives to be optimized, optional values: ['revenue', 'fairness', 'alignment']

    Returns:
        pd.DataFrame: the loaded DataFrame with `item_id` as index.
    """

    item_meta_morec = pd.read_csv(filepath, sep=",")
    col_names = item_meta_morec.columns
    assert "item_id" in col_names, "`item_id` column is required and all items are expected to be included."

    # check whether required columns are given
    err_info = "`{col}` information is missing in `item_meta_morec` file, which is required by {obj} objective."
    if "revenue" in objectives:
        assert "weight" in col_names, err_info.format(col="weight", obj="revenue")
    if "fairness" in objectives:
        assert "fair_group" in col_names, err_info.format(col="fair_group", obj="fairness")
    if "alignment" in objectives:
        assert "align_group" in col_names, err_info.format(col="align_group", obj="alignment")

    # check whether there are non-padding items missing
    items = item_meta_morec["item_id"].unique()
    if len(items) < n_items:
        if len(items) == (n_items - 1) and (0 not in items):
            # only padding item is misses, pad it into dataframe
            new_row = pd.DataFrame({"item_id": [0], "weight": [0.0], "fair_group": [0], "align_group": [0]})
            item_meta_morec = pd.concat((new_row, item_meta_morec), ignore_index=True)
        else:
            raise ValueError(f"There are `{n_items}` items in dataset but only `{len(items)}` items have meta information.")

    for col in ("align_group", "fair_group"):
        if item_meta_morec[col].min() == 0:
            _items = item_meta_morec[item_meta_morec[col] == 0]["item_id"].unique()
            if len(_items) > 1 or _items[0] != 0:  # if the group id range from 0 originally, increase 1 to leave 0 as padding index
                item_meta_morec.loc[item_meta_morec["item_id"] != 0, col] += 1

    item_meta_morec.set_index("item_id", drop=True, inplace=True)

    return item_meta_morec


def load_alignment_distribution(
    item2meta_morec: pd.DataFrame,
    item2popularity: np.ndarray,
    align_dist_filepath: str = None,
) -> np.ndarray:
    """
    Get/load the expected distribution for alignment objective.

    In term of the alignment objective, an expected distribution is required. Here two cases are considered:

    1. Align the recommendation to the distribution in training set. The distribution is calculated using the item's frequency in training set.
    2. Align the recommendation to a given distribution. The distribution is saved in a csv-format file, which consists of two columns: 'group_id' and
    'proportion'. The distribution is calculated by normalizing the proportion of each group.

    Args:
        item2meta_morec (pd.DataFrame): loaded item's meta information, item's align_group in it is required to get the distribution in case 1.
        item2popularity (np.ndarray): the item's frequency in training set, used for calculating distribution in case 1.
        align_dist_filepath (str): the path of the csv-format file where distribution of groups is saved, which is used to load expected distribution in case 2.

    Returns:
        np.ndarray: the normalized distribution of alignment group.
    """
    if "align_group" not in item2meta_morec.columns:
        return None
    max_group_id = item2meta_morec["align_group"].max()
    group2prob = np.zeros(max_group_id)
    if align_dist_filepath is not None:  # load from csv-format file
        exp_dist = pd.read_csv(align_dist_filepath, sep=",")
        columns = exp_dist.columns
        assert ("group_id" in columns) and (
            "proportion" in columns
        ), "`group_id` and `proportion` are required in the expected distribution file for alignment,"
        groups = exp_dist["group_id"].unique()
        assert len(groups) == exp_dist.shape[0], f"There are duplicated group ids in `{align_dist_filepath}`."
        # missing_group = [g for g in range(1, max_group_id)]
        group2prob[exp_dist["group_id"]] = exp_dist["proportion"]
    else:
        for gid in range(1, max_group_id + 1):
            _items = item2meta_morec[item2meta_morec["align_group"] == gid].index.to_numpy()
            group2prob[gid - 1] = item2popularity[_items].sum()
    group2prob = group2prob / (group2prob.sum() + 1e-10)
    return group2prob


__all__ = [
    "load_morec_meta_data",
    "load_alignment_distribution",
    "MoRecDS",
    "PIController",
    "PIPEMTLController",
    "ParetoSolver",
    "MGDASolver",
    "ParetoMTLSolver",
    "EPOSolver",
]
