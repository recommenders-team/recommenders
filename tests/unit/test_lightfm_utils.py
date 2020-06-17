# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import itertools
import numpy as np
import pandas as pd
import lightfm
from lightfm import LightFM, cross_validation
from lightfm.data import Dataset
from reco_utils.recommender.lightfm.lightfm_utils import (
    compare_metric,
    track_model_metrics,
    similar_users,
    similar_items,
)

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

    (interactions, _) = dataset.build_interactions(df.iloc[:, 0:3].values)

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


def test_interactions(interactions):
    train_interactions, test_interactions, item_features, user_features = interactions
    assert train_interactions.shape == (10, 10)
    assert test_interactions.shape == (10, 10)
    assert item_features.shape == (10, 19)
    assert user_features.shape == (10, 17)


def test_fitting(fitting):
    output, _ = fitting
    assert output.shape == (600, 4)


def test_sim_users(sim_users):
    assert sim_users.shape == (5, 2)


def test_sim_items(sim_items):
    assert sim_items.shape == (5, 2)
