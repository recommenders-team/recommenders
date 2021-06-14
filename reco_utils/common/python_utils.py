# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import logging
import numpy as np
from scipy import sparse


logger = logging.getLogger()


def exponential_decay(value, max_val, half_life):
    """Compute decay factor for a given value based on an exponential decay.

    Values greater than `max_val` will be set to 1.

    Args:
        value (numeric): Value to calculate decay factor
        max_val (numeric): Value at which decay factor will be 1
        half_life (numeric): Value at which decay factor will be 0.5

    Returns:
        float: Decay factor
    """
    return np.minimum(1.0, np.power(0.5, (max_val - value) / half_life))


def jaccard(cooccurrence):
    """Helper method to calculate the Jaccard similarity of a matrix of co-occurrences.

    Args:
        cooccurrence (numpy.ndarray): the symmetric matrix of co-occurrences of items.

    Returns:
        numpy.ndarray: The matrix of Jaccard similarities between any two items.
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cooccurrence / (diag_rows + diag_cols - cooccurrence)

    return np.array(result)


def lift(cooccurrence):
    """Helper method to calculate the Lift of a matrix of co-occurrences.

    Args:
        cooccurrence (numpy.ndarray): The symmetric matrix of co-occurrences of items.

    Returns:
        numpy.ndarray: The matrix of Lifts between any two items.
    """

    diag = cooccurrence.diagonal()
    diag_rows = np.expand_dims(diag, axis=0)
    diag_cols = np.expand_dims(diag, axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        result = cooccurrence / (diag_rows * diag_cols)

    return np.array(result)


def get_top_k_scored_items(scores, top_k, sort_top_k=False):
    """Extract top K items from a matrix of scores for each user-item pair, optionally sort results per user.

    Args:
        scores (numpy.ndarray): Score matrix (users x items).
        top_k (int): Number of top items to recommend.
        sort_top_k (bool): Flag to sort top k results.

    Returns:
        numpy.ndarray, numpy.ndarray:
        - Indices into score matrix for each users top items.
        - Scores corresponding to top items.

    """

    # ensure we're working with a dense ndarray
    if isinstance(scores, sparse.spmatrix):
        scores = scores.todense()

    if scores.shape[1] < top_k:
        logger.warning(
            "Number of items is less than top_k, limiting top_k to number of items"
        )
    k = min(top_k, scores.shape[1])

    test_user_idx = np.arange(scores.shape[0])[:, None]

    # get top K items and scores
    # this determines the un-ordered top-k item indices for each user
    top_items = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = scores[test_user_idx, top_items]

    if sort_top_k:
        sort_ind = np.argsort(-top_scores)
        top_items = top_items[test_user_idx, sort_ind]
        top_scores = top_scores[test_user_idx, sort_ind]

    return np.array(top_items), np.array(top_scores)


def binarize(a, threshold):
    """Binarize the values.

    Args:
        a (numpy.ndarray): Input array that needs to be binarized.
        threshold (float): Threshold below which all values are set to 0, else 1.

    Returns:
        numpy.ndarray: Binarized array.
    """
    return np.where(a > threshold, 1.0, 0.0)


def rescale(data, new_min=0, new_max=1, data_min=None, data_max=None):
    """Rescale/normalize the data to be within the range `[new_min, new_max]`
    If data_min and data_max are explicitly provided, they will be used
    as the old min/max values instead of taken from the data.

    .. note::
        This is same as the `scipy.MinMaxScaler` with the exception that we can override
        the min/max of the old scale.

    Args:
        data (numpy.ndarray): 1d scores vector or 2d score matrix (users x items).
        new_min (int|float): The minimum of the newly scaled data.
        new_max (int|float): The maximum of the newly scaled data.
        data_min (None|number): The minimum of the passed data [if omitted it will be inferred].
        data_max (None|number): The maximum of the passed data [if omitted it will be inferred].

    Returns:
        numpy.ndarray: The newly scaled/normalized data.
    """
    data_min = data.min() if data_min is None else data_min
    data_max = data.max() if data_max is None else data_max
    return (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min
