# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.sar import SAR


@pytest.fixture(
    params=[
        {},
        {
            "col_user": "person",
            "col_item": "product",
            "col_rating": "weight",
            "col_prediction": "score",
        },
    ],
    ids=["default_columns", "custom_columns"],
)
def fitted_sar(request):
    model = SAR(**request.param)
    train = pd.DataFrame(
        {
            model.col_user: [
                "first",
                "second",
                "second",
                "complete",
                "complete",
                "complete",
                "complete",
            ],
            model.col_item: [41, 41, 7, 41, 7, 93, 18],
            model.col_rating: [2.0, 1.0, 3.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    model.fit(train)
    return model


def test_score_remove_seen_respects_requested_user_order(fitted_sar):
    model = fitted_sar
    users = pd.DataFrame({model.col_user: ["second", "first", "second"]})
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
    users = pd.DataFrame({model.col_user: ["second", "first"]})
    before = model.score(users)
    affinity = model.user_affinity.copy()
    similarity = model.item_similarity.copy()

    first_masked = model.score(users, remove_seen=True)
    second_masked = model.score(users, remove_seen=True)

    np.testing.assert_array_equal(first_masked, second_masked)
    np.testing.assert_array_equal(model.score(users), before)
    assert (model.user_affinity != affinity).nnz == 0
    np.testing.assert_array_equal(model.item_similarity, similarity)


@pytest.mark.parametrize("top_k", [1, 3, 10])
@pytest.mark.parametrize("sort_top_k", [False, True])
def test_recommend_remove_seen_returns_only_available_candidates(
    fitted_sar, top_k, sort_top_k
):
    model = fitted_sar
    users = pd.DataFrame({model.col_user: ["second", "complete", "first", "second"]})
    unseen = {"first": {7, 93, 18}, "second": {93, 18}, "complete": set()}

    recommendations = model.recommend_k_items(
        users, top_k=top_k, sort_top_k=sort_top_k, remove_seen=True
    )

    assert np.isfinite(recommendations[model.col_prediction]).all()
    assert not recommendations.duplicated([model.col_user, model.col_item]).any()
    for user, candidates in unseen.items():
        rows = recommendations[recommendations[model.col_user] == user]
        assert len(rows) == min(top_k, len(candidates))
        assert set(rows[model.col_item]) <= candidates
        if top_k >= len(candidates):
            assert set(rows[model.col_item]) == candidates
        if sort_top_k:
            assert rows[model.col_prediction].is_monotonic_decreasing


def test_recommend_remove_seen_returns_empty_for_fully_seen_catalog(fitted_sar):
    model = fitted_sar
    users = pd.DataFrame({model.col_user: ["complete"]})

    recommendations = fitted_sar.recommend_k_items(users, top_k=10, remove_seen=True)

    assert recommendations.empty
    assert list(recommendations.columns) == [
        model.col_user,
        model.col_item,
        model.col_prediction,
    ]


def test_remove_seen_is_opt_in(fitted_sar):
    model = fitted_sar
    users = pd.DataFrame({model.col_user: ["complete"]})
    original_users = users.copy(deep=True)

    default = model.recommend_k_items(users, top_k=10, sort_top_k=True)
    explicit = model.recommend_k_items(
        users, top_k=10, sort_top_k=True, remove_seen=False
    )

    pd.testing.assert_frame_equal(default, explicit)
    assert set(default[model.col_item]) == set(model.item2index)
    assert np.isfinite(default[model.col_prediction]).all()
    np.testing.assert_array_equal(
        model.score(users), model.score(users, remove_seen=False)
    )
    model.recommend_k_items(users, top_k=10, remove_seen=True)
    pd.testing.assert_frame_equal(users, original_users)
