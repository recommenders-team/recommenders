# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    from recommenders.models.two_tower.data_loader import (
        InteractionDataset,
        TwoTowerData,
    )
    from recommenders.models.two_tower.model import TwoTowerModel
    from recommenders.models.two_tower.utils import (
        multi_hot_features,
        recommend_k_items,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment

from recommenders.utils.constants import (
    DEFAULT_GENRE_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)


@pytest.fixture
def interactions():
    return pd.DataFrame(
        {
            DEFAULT_USER_COL: ["a", "a", "b", "b", "b", "c"],
            DEFAULT_ITEM_COL: [10, 20, 20, 30, 30, 40],
            DEFAULT_RATING_COL: [5, 4, 3, 5, 5, 1],
        }
    )


@pytest.fixture
def item_genres():
    return pd.DataFrame(
        {
            DEFAULT_ITEM_COL: [10, 20, 30, 99],
            DEFAULT_GENRE_COL: ["Action|Comedy", "Drama", "Action", "Horror"],
        }
    )


@pytest.mark.gpu
def test_interaction_dataset():
    ds = InteractionDataset(np.array([0, 1, 2]), np.array([3, 4, 5]))
    assert len(ds) == 3
    user, item = ds[1]
    assert user.item() == 1 and item.item() == 4
    assert user.dtype == torch.long
    with pytest.raises(ValueError):
        InteractionDataset(np.array([0, 1]), np.array([0]))


@pytest.mark.gpu
def test_two_tower_data_indexing(interactions):
    data = TwoTowerData(interactions)
    assert data.n_users == 3
    assert data.n_items == 4
    assert list(data.users) == ["a", "b", "c"]
    assert list(data.items) == [10, 20, 30, 40]
    assert data.user2idx == {"a": 0, "b": 1, "c": 2}
    np.testing.assert_array_equal(data.item_counts, [1, 2, 2, 1])
    # Duplicated interactions are counted for the logQ correction, but
    # only flagged once as seen
    assert data.seen.shape == (3, 4)
    assert data.seen.nnz == 5
    assert data.user_features is None
    assert data.item_features is None


@pytest.mark.gpu
def test_two_tower_data_invalid_input(interactions):
    with pytest.raises(ValueError):
        TwoTowerData(interactions.iloc[0:0])
    with pytest.raises(ValueError):
        TwoTowerData(interactions, col_user="missing")
    with pytest.raises(ValueError):
        TwoTowerData(interactions, item_features=pd.DataFrame({"x": [1.0]}))
    with pytest.raises(ValueError):
        TwoTowerData(interactions, item_features=pd.DataFrame({DEFAULT_ITEM_COL: [10]}))


@pytest.mark.gpu
def test_two_tower_data_features(interactions, item_genres):
    features = multi_hot_features(item_genres)
    data = TwoTowerData(interactions, item_features=features)
    # Columns are sorted alphabetically: Action, Comedy, Drama, Horror
    assert data.item_features.shape == (4, 4)
    assert data.item_features.dtype == np.float32
    np.testing.assert_array_equal(data.item_features[0], [1, 1, 0, 0])  # item 10
    np.testing.assert_array_equal(data.item_features[2], [1, 0, 0, 0])  # item 30
    # Item 40 has no genres and item 99 is not in the training data
    np.testing.assert_array_equal(data.item_features[3], [0, 0, 0, 0])


@pytest.mark.gpu
def test_two_tower_data_loader(interactions):
    data = TwoTowerData(interactions)
    dl = data.loader(batch_size=100, shuffle=False)
    users, items = next(iter(dl))
    # Batch size is capped at the number of interactions
    assert len(users) == 6
    np.testing.assert_array_equal(users.numpy(), [0, 0, 1, 1, 1, 2])
    np.testing.assert_array_equal(items.numpy(), [0, 1, 1, 2, 2, 3])

    order_1 = [u.tolist() for u, _ in data.loader(batch_size=2, seed=1)]
    order_2 = [u.tolist() for u, _ in data.loader(batch_size=2, seed=1)]
    assert order_1 == order_2

    other = pd.DataFrame(
        {DEFAULT_USER_COL: ["a", "z", "c"], DEFAULT_ITEM_COL: [40, 10, 77]}
    )
    users, items = next(iter(data.loader(other, shuffle=False)))
    # Only ("a", 40) has a known user and a known item
    assert users.tolist() == [0] and items.tolist() == [3]

    with pytest.raises(ValueError):
        data.loader(pd.DataFrame({DEFAULT_USER_COL: ["z"], DEFAULT_ITEM_COL: [1]}))


@pytest.mark.gpu
def test_multi_hot_features(item_genres):
    features = multi_hot_features(item_genres, prefix="genre_")
    assert list(features.columns) == [
        DEFAULT_ITEM_COL,
        "genre_Action",
        "genre_Comedy",
        "genre_Drama",
        "genre_Horror",
    ]
    assert len(features) == 4
    assert features.iloc[0, 1:].tolist() == [1.0, 1.0, 0.0, 0.0]


@pytest.mark.gpu
def test_recommend_k_items(interactions):
    torch.manual_seed(0)
    data = TwoTowerData(interactions)
    model = TwoTowerModel(n_users=data.n_users, n_items=data.n_items)

    recs = recommend_k_items(model, data, top_k=2)
    assert list(recs.columns) == [
        DEFAULT_USER_COL,
        DEFAULT_ITEM_COL,
        DEFAULT_PREDICTION_COL,
    ]
    assert recs.groupby(DEFAULT_USER_COL).size().to_dict() == {"a": 2, "b": 2, "c": 2}

    # Seen items are removed
    seen = set(zip(interactions[DEFAULT_USER_COL], interactions[DEFAULT_ITEM_COL]))
    recommended = set(zip(recs[DEFAULT_USER_COL], recs[DEFAULT_ITEM_COL]))
    assert seen.isdisjoint(recommended)

    # Scores are sorted in descending order within each user
    for _, group in recs.groupby(DEFAULT_USER_COL):
        assert group[DEFAULT_PREDICTION_COL].is_monotonic_decreasing


@pytest.mark.gpu
def test_recommend_k_items_edge_cases(interactions):
    data = TwoTowerData(interactions)
    model = TwoTowerModel(n_users=data.n_users, n_items=data.n_items)

    # top_k larger than the catalogue, seen items kept
    recs = recommend_k_items(model, data, users=["a"], top_k=10, remove_seen=False)
    assert len(recs) == data.n_items

    # User "b" has seen 2 of the 4 items, so only 2 remain
    recs = recommend_k_items(model, data, users=["b"], top_k=10)
    assert sorted(recs[DEFAULT_ITEM_COL]) == [10, 40]

    # Unknown users are skipped
    recs = recommend_k_items(model, data, users=["a", "unknown"], top_k=1)
    assert recs[DEFAULT_USER_COL].tolist() == ["a"]
    recs = recommend_k_items(model, data, users=["unknown"], top_k=1)
    assert len(recs) == 0

    with pytest.raises(ValueError):
        recommend_k_items(model, data, top_k=0)
