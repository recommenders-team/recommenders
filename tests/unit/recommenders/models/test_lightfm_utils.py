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
@pytest.mark.parametrize("entity", ["users", "items"])
@pytest.mark.parametrize("feature_kind", ["shared", "zero", "distinct"])
@pytest.mark.parametrize("count", [0, 1, 3, 5])
def test_similar_entities_exclude_query_with_ties(entity, feature_kind, count):
    """Return other entities even when the query is tied with several candidates."""
    from scipy import sparse

    if feature_kind == "distinct":
        features = sparse.eye(6, dtype=np.float32, format="csr")
    elif feature_kind == "shared":
        features = sparse.csr_matrix(np.ones((6, 1), dtype=np.float32))
    else:
        features = sparse.csr_matrix((6, 1), dtype=np.float32)

    training_features = (
        sparse.csr_matrix(np.ones((6, 1), dtype=np.float32))
        if feature_kind == "zero"
        else features
    )
    fitted = LightFM(no_components=4, loss="warp", random_state=7).fit(
        sparse.eye(6, dtype=np.float32, format="coo"),
        user_features=training_features,
        item_features=training_features,
        epochs=1,
        num_threads=1,
    )
    if entity == "users":
        search = similar_users
        id_column = "userID"
        _, representations = fitted.get_user_representations(features)
    else:
        search = similar_items
        id_column = "itemID"
        _, representations = fitted.get_item_representations(features)

    # Normalize in float64 independently of the helper's ranking implementation.
    vectors = representations.astype(np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms == 0, 1, norms)
    cosine = normalized @ normalized.T
    for query in range(6):
        result = search(query, features, fitted, N=count)
        ids = result[id_column].to_numpy(dtype=int)
        scores = result["score"].to_numpy(dtype=float)
        assert len(result) == count
        assert len(set(ids)) == count
        assert query not in ids
        assert result["score"].is_monotonic_decreasing
        np.testing.assert_allclose(scores, cosine[query, ids], rtol=1e-6, atol=1e-7)
        expected_scores = sorted(
            (cosine[query, other] for other in range(6) if other != query),
            reverse=True,
        )[:count]
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-6, atol=1e-7)
