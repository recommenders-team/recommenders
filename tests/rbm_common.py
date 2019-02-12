# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pytest

from reco_utils.dataset.python_splitters import numpy_stratified_split


@pytest.fixture(scope="module")
def test_specs():
    return {
        "users": 30,
        "items": 53,
        "ratings": 5,
        "seed": 123,
        "spars": 0.8,
        "ratio": 0.7,
    }


@pytest.fixture(scope="module")
def affinity_matrix(test_specs):
    """Generate a random user/item affinity matrix. By increasing the likehood of 0 elements we simulate
    a typical recommending situation where the input matrix is highly sparse.

    Args:
        users (int): number of users (rows).
        items (int): number of items (columns).
        ratings (int): rating scale, e.g. 5 meaning rates are from 1 to 5.
        spars: probability of obtaining zero. This roughly corresponds to the sparseness.
               of the generated matrix. If spars = 0 then the affinity matrix is dense.

    Returns:
        np.array: sparse user/affinity matrix of integers.

    """

    np.random.seed(test_specs["seed"])

    # uniform probability for the 5 ratings
    s = [(1 - test_specs["spars"]) / test_specs["ratings"]] * test_specs["ratings"]
    s.append(test_specs["spars"])
    P = s[::-1]

    # generates the user/item affinity matrix. Ratings are from 1 to 5, with 0s denoting unrated items
    X = np.random.choice(
        test_specs["ratings"] + 1, (test_specs["users"], test_specs["items"]), p=P
    )

    Xtr, Xtst = numpy_stratified_split(
        X, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    return (Xtr, Xtst)
