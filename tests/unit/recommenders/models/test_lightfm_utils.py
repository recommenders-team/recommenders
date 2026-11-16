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
@pytest.mark.parametrize(
    "with_users,with_items", itertools.product([False, True], repeat=2)
)
@pytest.mark.parametrize("num_threads", [1, 2])
@pytest.mark.parametrize("sparse_format", ["coo", "csr"])
def test_prepare_all_predictions_native_parity(
    with_users, with_items, num_threads, sparse_format
):
    """Keep scores, external IDs, pair order and weighted-interaction filtering."""
    from scipy import sparse
    from recommenders.models.lightfm.lightfm_utils import prepare_all_predictions

    data = pd.DataFrame(
        {
            "userID": ["u_m", "u_z", "u_a", "u_m", "u_z"],
            "itemID": [17, 23, 11, 5, 17],
        }
    )
    original = data.copy(deep=True)
    uid_map = {"u_z": 0, "u_a": 1, "u_m": 2}
    iid_map = {23: 0, 5: 1, 17: 2, 11: 3}
    # Include duplicate COO coordinates, fractional positives and a negative entry.
    interactions = sparse.coo_matrix(
        (
            np.array([0.4, 0.6, -1, 0.5, 1.5, 1, 0.25], dtype=np.float32),
            ([0, 0, 0, 0, 1, 2, 2], [0, 0, 2, 3, 1, 2, 3]),
        ),
        shape=(3, 4),
    ).asformat(sparse_format)
    dense = interactions.toarray()
    user_features = (
        sparse.csr_matrix([[1, 0, 0.5], [0.2, 1, 0], [0, 0.3, 1]], dtype=np.float32)
        if with_users
        else None
    )
    item_features = (
        sparse.csr_matrix([[1, 0], [0, 1], [0.5, 1], [1, 0.5]], dtype=np.float32)
        if with_items
        else None
    )
    fitted = LightFM(no_components=4, loss="warp", random_state=7).fit(
        sparse.coo_matrix((dense > 0).astype(np.float32)),
        user_features=user_features,
        item_features=item_features,
        epochs=2,
        num_threads=1,
    )
    rows = []
    for user in data.userID.unique():
        for item in data.itemID.unique():
            uid, iid = uid_map[user], iid_map[item]
            if dense[uid, iid] < 1:
                prediction = fitted.predict(
                    np.array([uid], dtype=np.int32),
                    np.array([iid], dtype=np.int32),
                    user_features=user_features,
                    item_features=item_features,
                    num_threads=num_threads,
                )[0]
                rows.append((user, item, prediction))
    expected = pd.DataFrame(rows, columns=["userID", "itemID", "prediction"])
    result = prepare_all_predictions(
        data,
        uid_map,
        iid_map,
        interactions,
        fitted,
        num_threads,
        user_features=user_features,
        item_features=item_features,
    )
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_array_equal(interactions.toarray(), dense)


@pytest.mark.experimental
@pytest.mark.parametrize("empty_kind", ["empty_data", "all_seen", "single_seen"])
@pytest.mark.parametrize("with_features", [False, True])
def test_prepare_all_predictions_empty_selection(empty_kind, with_features):
    """An evaluation selection with no unseen pairs has a valid empty schema."""
    from scipy import sparse
    from recommenders.models.lightfm.lightfm_utils import prepare_all_predictions

    data = pd.DataFrame({"userID": ["user_a", "user_b"], "itemID": [23, 17]})
    if empty_kind == "empty_data":
        data = data.iloc[:0]
    elif empty_kind == "single_seen":
        data = data.iloc[:1]
    interactions = sparse.coo_matrix(np.ones((2, 2), dtype=np.float32))
    features = sparse.eye(2, dtype=np.float32, format="csr") if with_features else None
    fitted = LightFM(no_components=4, loss="logistic", random_state=7).fit(
        interactions,
        user_features=features,
        item_features=features,
        epochs=1,
        num_threads=1,
    )
    result = prepare_all_predictions(
        data,
        {"user_a": 0, "user_b": 1},
        {23: 0, 17: 1},
        interactions,
        fitted,
        1,
        user_features=features,
        item_features=features,
    )
    assert result.empty
    assert result.columns.tolist() == ["userID", "itemID", "prediction"]
    assert result["prediction"].dtype == np.float32
