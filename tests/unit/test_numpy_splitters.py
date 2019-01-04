# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import numpy as np
import pytest

from reco_utils.dataset.numpy_splitters import numpy_stratified_split


@pytest.fixture(scope="module")
def test_specs():

    return {
        "number_of_items": 50,
        "number_of_users": 20,
        "seed": 123,
        "ratio": 0.6,
        "tolerance": 0.01,
        "fluctuation": 0.02,
    }


@pytest.fixture(scope="module")
def python_dataset(test_specs):

    """Generate a test user/item affinity Matrix"""

    # fix the the random seed
    np.random.seed(test_specs["seed"])

    # generates the user/item affinity matrix. Ratings are from 1 to 5, with 0s denoting unrated items
    X = np.random.randint(
        low=0,
        high=6,
        size=(test_specs["number_of_users"], test_specs["number_of_items"]),
    )

    return X


def test_random_stratified_splitter(test_specs, python_dataset):
    """
    Test the random stratified splitter.
    """

    # generate a syntetic dataset
    X = python_dataset

    # the splitter returns (in order): train and test user/affinity matrices, train and test datafarmes and user/items to matrix maps
    Xtr, Xtst = numpy_stratified_split(
        X, ratio=test_specs["ratio"], seed=test_specs["seed"]
    )

    # Tests
    # check that the generated matrices have the correct dimensions
    assert (Xtr.shape[0] == X.shape[0]) & (Xtr.shape[1] == X.shape[1])

    assert (Xtst.shape[0] == X.shape[0]) & (Xtst.shape[1] == X.shape[1])

    X_rated = np.sum(X != 0, axis=1)  # number of total rated items per user
    Xtr_rated = np.sum(Xtr != 0, axis=1)  # number of rated items in the train set
    Xtst_rated = np.sum(Xtst != 0, axis=1)  # number of rated items in the test set

    # global split: check that the all dataset is split in the correct ratio
    assert (
        Xtr_rated.sum() / (X_rated.sum())
        <= test_specs["ratio"] + test_specs["tolerance"]
    ) & (
        Xtr_rated.sum() / (X_rated.sum())
        >= test_specs["ratio"] - test_specs["tolerance"]
    )

    assert (
        Xtst_rated.sum() / (X_rated.sum())
        <= (1 - test_specs["ratio"]) + test_specs["tolerance"]
    ) & (
        Xtr_rated.sum() / (X_rated.sum())
        >= (1 - test_specs["ratio"]) - test_specs["tolerance"]
    )

    # This implementation of the stratified splitter performs a random split at the single user level. Here we check
    # that also this more stringent condition is verified. Note that user to user fluctuations in the split ratio
    # are stronger than for the entire dataset due to the random nature of the per user splitting.
    # For this reason we allow a slightly bigger tollerance, as specified in the test_specs()

    assert (
        (Xtr_rated / X_rated <= test_specs["ratio"] + test_specs["fluctuation"]).all()
        & (Xtr_rated / X_rated >= test_specs["ratio"] - test_specs["fluctuation"]).all()
    )

    assert (
        (
            Xtst_rated / X_rated
            <= (1 - test_specs["ratio"]) + test_specs["fluctuation"]
        ).all()
        & (
            Xtst_rated / X_rated
            >= (1 - test_specs["ratio"]) - test_specs["fluctuation"]
        ).all()
    )
