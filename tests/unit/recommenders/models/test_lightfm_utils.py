# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import itertools
import numpy as np
import pandas as pd

try:
    from lightfm.data import Dataset
    from lightfm import LightFM, cross_validation
    from recommenders.models.lightfm.lightfm_utils import (
        track_model_metrics,
        similar_users,
        similar_items,
    )
except ModuleNotFoundError:
    pass


SEEDNO = 42
TEST_PERCENTAGE = 0.25
TEST_USER_ID = 2
TEST_ITEM_ID = 1


# note user and item ID need to be sequential for similar users and similar items to work
@pytest.fixture(scope="module")
def df():
    mock_data = {
        "userID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "itemID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "rating": [2.0, 4.0, 1.0, 4.0, 1.0, 2.0, 5.0, 1.0, 1.0, 2.0],
        "genre": [
            "Action|Comedy",
            "Drama",
            "Drama|Romance|War",
            "Drama|Sci-Fi",
            "Horror",
            "Action|Horror|Sci-Fi|Thriller",
            "Drama|Romance|War",
            "Western",
            "Comedy",
            "Horror",
        ],
        "occupation": [
            "engineer",
            "student",
            "retired",
            "administrator",
            "writer",
            "administrator",
            "student",
            "executive",
            "student",
            "other",
        ],
    }
    return pd.DataFrame(mock_data)


@pytest.fixture(scope="module")
def interactions(df):
    movie_genre = [x.split("|") for x in df["genre"]]
    all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))

    all_occupations = sorted(list(set(df["occupation"])))

    dataset = Dataset()
    dataset.fit(
        df["userID"],
        df["itemID"],
        item_features=all_movie_genre,
        user_features=all_occupations,
    )

    item_features = dataset.build_item_features(
        (x, y) for x, y in zip(df.itemID, movie_genre)
    )

    user_features = dataset.build_user_features(
        (x, [y]) for x, y in zip(df.userID, df["occupation"])
    )

    interactions, _ = dataset.build_interactions(df.iloc[:, 0:3].values)

    train_interactions, test_interactions = cross_validation.random_train_test_split(
        interactions,
        test_percentage=TEST_PERCENTAGE,
        random_state=np.random.RandomState(SEEDNO),
    )
    return train_interactions, test_interactions, item_features, user_features


@pytest.fixture(scope="module")
def model():
    return LightFM(loss="warp", random_state=np.random.RandomState(SEEDNO))


@pytest.fixture(scope="module")
def fitting(model, interactions, df):
    train_interactions, test_interactions, item_features, user_features = interactions
    output, fitted_model = track_model_metrics(
        model=model,
        train_interactions=train_interactions,
        test_interactions=test_interactions,
        user_features=user_features,
        item_features=item_features,
        no_epochs=1,
        show_plot=False,
    )
    return output, fitted_model


@pytest.fixture(scope="module")
def sim_users(interactions, fitting):
    _, _, _, user_features = interactions
    _, fitted_model = fitting
    return similar_users(
        user_id=TEST_USER_ID, user_features=user_features, model=fitted_model, N=5
    )


@pytest.fixture(scope="module")
def sim_items(interactions, fitting):
    _, _, item_features, _ = interactions
    _, fitted_model = fitting
    return similar_items(
        item_id=TEST_ITEM_ID, item_features=item_features, model=fitted_model, N=5
    )


@pytest.mark.experimental
def test_interactions(interactions):
    train_interactions, test_interactions, item_features, user_features = interactions
    assert train_interactions.shape == (10, 10)
    assert test_interactions.shape == (10, 10)
    assert item_features.shape == (10, 19)
    assert user_features.shape == (10, 17)


@pytest.mark.experimental
@pytest.mark.skip(reason="Flaky test")
def test_fitting(fitting):
    output, _ = fitting
    assert output.shape == (4, 4)
    target = np.array(
        [
            [0, 0.10000000894069672, "train", "Precision"],
            [0, 0.10000000149011612, "test", "Precision"],
            [0, 1.0, "train", "Recall"],
            [0, 1.0, "test", "Recall"],
        ],
        dtype="object",
    )
    np.testing.assert_array_equal(output, target)


@pytest.mark.experimental
def test_sim_users(sim_users):
    assert sim_users.shape == (5, 2)


@pytest.mark.experimental
def test_sim_items(sim_items):
    assert sim_items.shape == (5, 2)


@pytest.mark.experimental
def test_prepare_test_df():
    try:
        from scipy import sparse
        from recommenders.models.lightfm.lightfm_utils import prepare_test_df
    except ModuleNotFoundError:
        pytest.skip("lightfm or scipy not installed")
        return

    uids = np.array([0, 1, 2, 0, 1])
    iids = np.array([0, 1, 2, 1, 0])
    uid_map = {"user_a": 0, "user_b": 1, "user_c": 2}
    iid_map = {"item_x": 0, "item_y": 1, "item_z": 2}

    weights_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    weights = sparse.coo_matrix((weights_data, (uids, iids)), shape=(3, 3))

    test_idx = slice(0, 3)
    result = prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights)

    assert result.shape[0] == 3, "Should have 3 rows"
    assert list(result.columns) == [
        "userID",
        "itemID",
        "rating",
    ], "Should have correct columns"
    assert result["userID"].tolist() == [
        "user_a",
        "user_b",
        "user_c",
    ], "User IDs should be mapped correctly"
    assert result["itemID"].tolist() == [
        "item_x",
        "item_y",
        "item_z",
    ], "Item IDs should be mapped correctly"
    assert result["rating"].tolist() == [1.0, 2.0, 3.0], "Ratings should be correct"


@pytest.mark.experimental
@pytest.mark.parametrize("id_kind", ["string", "integer", "mixed"])
@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("selection", ["all", "slice", "repeated", "boolean", "empty"])
def test_prepare_test_df_explicit_mapping_values(id_kind, reordered, selection):
    """External evaluation IDs depend on mapping values, not insertion order."""
    from recommenders.models.lightfm.lightfm_utils import prepare_test_df

    if id_kind == "string":
        users, items = ["user_z", "user_a", "user_m"], ["item_q", "item_c", "item_n"]
    elif id_kind == "integer":
        users, items = [23, 5, 17], [2**60 + 7, 2**60 + 1, 2**60 + 3]
    else:
        users, items = [42, "42", 2**60 + 7], ["item_q", 29, "item_n"]

    records = [
        (users[0], items[0], 1.0),
        (users[1], items[1], 2.0),
        (users[2], items[2], 3.0),
        (users[0], items[1], 4.0),
        (users[2], items[0], 5.0),
    ]
    dataset = Dataset()
    dataset.fit(users, items)
    _, weights = dataset.build_interactions(records)
    uid_map, _, iid_map, _ = dataset.mapping()
    if reordered:
        uid_map = dict(reversed(list(uid_map.items())))
        iid_map = {items[index]: iid_map[items[index]] for index in [1, 2, 0]}
    original_users, original_items = list(uid_map.items()), list(iid_map.items())
    test_idx = {
        "all": slice(None),
        "slice": slice(1, 5, 2),
        "repeated": np.array([4, 1, 4, 0]),
        "boolean": np.array([True, False, True, False, True]),
        "empty": slice(0, 0),
    }[selection]
    expected = pd.DataFrame(records, columns=["userID", "itemID", "rating"])
    expected["rating"] = expected["rating"].astype(np.float32)
    expected = expected.iloc[np.arange(len(records))[test_idx]].reset_index(drop=True)
    result = prepare_test_df(
        test_idx, weights.row, weights.col, uid_map, iid_map, weights
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    assert list(uid_map.items()) == original_users
    assert list(iid_map.items()) == original_items
