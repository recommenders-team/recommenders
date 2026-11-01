# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.sar import SAR


@pytest.fixture
def fitted_sar():
    train = pd.DataFrame(
        {
            "userID": [
                "first",
                "second",
                "second",
                "complete",
                "complete",
                "complete",
                "complete",
            ],
            "itemID": [41, 41, 7, 41, 7, 93, 18],
            "rating": [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    model = SAR()
    model.fit(train)
    return model


def test_score_remove_seen_respects_requested_user_order(fitted_sar):
    model = fitted_sar
    users = pd.DataFrame({"userID": ["second", "first", "second"]})
    unmasked = model.score(users)

    masked = np.asarray(model.score(users, remove_seen=True))

    expected_mask = np.zeros_like(unmasked, dtype=bool)
    expected_mask[0, [model.item2index[41], model.item2index[7]]] = True
    expected_mask[1, model.item2index[41]] = True
    assert masked.shape == (2, 4)
    np.testing.assert_array_equal(np.isneginf(masked), expected_mask)
    np.testing.assert_array_equal(masked[~expected_mask], unmasked[~expected_mask])


def test_score_remove_seen_does_not_change_later_scores(fitted_sar):
    model = fitted_sar
    users = pd.DataFrame({"userID": ["second", "first"]})
    before = model.score(users)
    affinity = model.user_affinity.copy()
    similarity = model.item_similarity.copy()

    first_masked = model.score(users, remove_seen=True)
    second_masked = model.score(users, remove_seen=True)

    np.testing.assert_array_equal(first_masked, second_masked)
    np.testing.assert_array_equal(model.score(users), before)
    assert (model.user_affinity != affinity).nnz == 0
    np.testing.assert_array_equal(model.item_similarity, similarity)
