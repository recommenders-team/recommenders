# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""Column names must propagate through every round of bipartite k-core peeling."""

import pandas as pd
import pytest
from recommenders.datasets.split_utils import filter_k_core


def interactions(user="userID", item="itemID"):
    return pd.DataFrame(
        {
            user: ["a", "a", "b", "b", "c", "c", "d", "e"],
            item: ["p", "q", "p", "q", "q", "r", "r", "s"],
            "rating": list(range(8)),
        }
    )


def peel_reference(frame, k, user, item):
    result = frame.copy()
    while True:
        old = len(result)
        users = result[user].value_counts()
        items = result[item].value_counts()
        result = result[result[user].map(users).ge(k) & result[item].map(items).ge(k)]
        if len(result) == old:
            return result.sort_values(user)


@pytest.mark.parametrize(
    "user,item", [("uid", "iid"), ("uid", "itemID"), ("userID", "iid")]
)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_custom_columns_match_independent_graph_peeling(user, item, k):
    frame = interactions(user, item)
    original = frame.copy(deep=True)
    actual = filter_k_core(frame, core_num=k, col_user=user, col_item=item)
    expected = peel_reference(frame, k, user, item)
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)


def test_default_schema_control():
    frame = interactions()
    pd.testing.assert_frame_equal(
        filter_k_core(frame, core_num=2), peel_reference(frame, 2, "userID", "itemID")
    )


def test_zero_core_custom_schema_control():
    frame = interactions("uid", "iid")
    pd.testing.assert_frame_equal(
        filter_k_core(frame, core_num=0, col_user="uid", col_item="iid"),
        frame.sort_values("uid"),
    )


def test_empty_custom_schema():
    frame = interactions("uid", "iid").iloc[:0]
    actual = filter_k_core(frame, core_num=2, col_user="uid", col_item="iid")
    pd.testing.assert_frame_equal(actual, frame)
