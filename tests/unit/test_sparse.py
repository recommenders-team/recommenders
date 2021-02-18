# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import pytest

from reco_utils.dataset.sparse import AffinityMatrix
from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)


@pytest.fixture(scope="module")
def test_specs():
    return {"number_of_items": 50, "number_of_users": 20, "seed": 123}


# generate a syntetic dataset
@pytest.fixture(scope="module")
def python_dataset(test_specs):

    """Get Python labels"""

    def random_date_generator(start_date, range_in_days):
        """Helper function to generate random timestamps.

        Reference: https://stackoverflow.com/questions/41006182/generate-random-dates-within-a
        -range-in-numpy

        """

        days_to_add = np.arange(0, range_in_days)
        random_dates = []

        for i in range(range_in_days):
            random_date = np.datetime64(start_date) + np.random.choice(days_to_add)
            random_dates.append(random_date)

        return random_dates

    # fix the the random seed
    np.random.seed(test_specs["seed"])

    # generates the user/item affinity matrix. Ratings are from 1 to 5, with 0s denoting unrated items
    X = np.random.randint(
        low=0,
        high=6,
        size=(test_specs["number_of_users"], test_specs["number_of_items"]),
    )

    # In the main code, input data are passed as pandas dataframe. Below we generate such df from the above matrix
    userids = []

    for i in range(1, test_specs["number_of_users"] + 1):
        userids.extend([i] * test_specs["number_of_items"])

    itemids = [i for i in range(1, test_specs["number_of_items"] + 1)] * test_specs[
        "number_of_users"
    ]
    ratings = np.reshape(X, -1)

    # create dataframe
    results = pd.DataFrame.from_dict(
        {
            DEFAULT_USER_COL: userids,
            DEFAULT_ITEM_COL: itemids,
            DEFAULT_RATING_COL: ratings,
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01",
                test_specs["number_of_users"] * test_specs["number_of_items"],
            ),
        }
    )

    # here we eliminate the missing ratings to obtain a standard form of the df as that of real data.
    results = results[results.rating != 0]

    return results


def test_df_to_sparse(test_specs, python_dataset):
    # initialize the splitter
    header = {
        "col_user": DEFAULT_USER_COL,
        "col_item": DEFAULT_ITEM_COL,
        "col_rating": DEFAULT_RATING_COL,
    }

    # instantiate the affinity matrix
    am = AffinityMatrix(DF=python_dataset, **header)

    # obtain the sparse matrix representation of the input dataframe
    X, _, _ = am.gen_affinity_matrix()

    # check that the generated matrix has the correct dimensions
    assert (X.shape[0] == python_dataset.userID.unique().shape[0]) & (
        X.shape[1] == python_dataset.itemID.unique().shape[0]
    )


def test_sparse_to_df(test_specs, python_dataset):
    # initialize the splitter
    header = {
        "col_user": DEFAULT_USER_COL,
        "col_item": DEFAULT_ITEM_COL,
        "col_rating": DEFAULT_RATING_COL,
    }

    # instantiate the the affinity matrix
    am = AffinityMatrix(DF=python_dataset, **header)

    # generate the sparse matrix representation
    X, _, _ = am.gen_affinity_matrix()

    # use the inverse function to generate a pandas df from a sparse matrix ordered by userID
    DF = am.map_back_sparse(X, kind="ratings")

    # tests: check that the two dataframes have the same elements in the same positions.
    assert (
        DF.userID.values.all()
        == python_dataset.sort_values(by=["userID"]).userID.values.all()
    )

    assert (
        DF.itemID.values.all()
        == python_dataset.sort_values(by=["userID"]).itemID.values.all()
    )

    assert (
        DF.rating.values.all()
        == python_dataset.sort_values(by=["userID"]).rating.values.all()
    )
