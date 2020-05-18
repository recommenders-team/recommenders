# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_TIMESTAMP_COL,
)
from reco_utils.dataset.python_splitters import python_chrono_split

# ncf data generation
@pytest.fixture(scope="module")
def test_specs_ncf():
    return {
        "number_of_rows": 1000,
        "user_ids": [1, 2, 3, 4, 5],
        "seed": 123,
        "ratio": 0.6,
        "split_numbers": [2, 3, 5],
        "tolerance": 0.01,
    }


@pytest.fixture(scope="module")
def python_dataset_ncf(test_specs_ncf):
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

    np.random.seed(test_specs_ncf["seed"])

    rating = pd.DataFrame(
        {
            DEFAULT_USER_COL: np.random.randint(
                1, 100, test_specs_ncf["number_of_rows"]
            ),
            DEFAULT_ITEM_COL: np.random.randint(
                1, 100, test_specs_ncf["number_of_rows"]
            ),
            DEFAULT_RATING_COL: np.random.randint(
                1, 5, test_specs_ncf["number_of_rows"]
            ),
            DEFAULT_TIMESTAMP_COL: random_date_generator(
                "2018-01-01", test_specs_ncf["number_of_rows"]
            ),
        }
    )

    train, test = python_chrono_split(rating, ratio=test_specs_ncf["ratio"])

    return train, test
