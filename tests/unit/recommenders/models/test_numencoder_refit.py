# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.lightgbm.lightgbm_utils import NumEncoder


def make_encoder(columns):
    return NumEncoder(columns, ["value"], "label", threshold=0, thresrate=1)


def training_frame(columns, replacement=False):
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0], "label": [0, 1, 0, 1]})
    for column in columns:
        frame[column] = ["a", "b", "c", "a"]
    if replacement:
        frame = frame.iloc[[3, 1, 0]].copy()
        frame["value"] = [12.0, 14.0, np.nan]
        frame["label"] = [1, 0, 1]
        for column in columns:
            frame[column] = ["new", "other", "new"]
    return frame


@pytest.mark.parametrize("columns", [["category"], ["genre", "region"]])
def test_repeated_numencoder_fit_preserves_features(columns):
    frame = training_frame(columns)
    encoder = make_encoder(columns)
    expected_x, expected_y = encoder.fit_transform(frame)

    for _ in range(2):
        actual_x, actual_y = encoder.fit_transform(frame)
        np.testing.assert_array_equal(actual_x, expected_x)
        np.testing.assert_array_equal(actual_y, expected_y)


@pytest.mark.parametrize("columns", [["category"], ["genre", "region"]])
def test_numencoder_refit_matches_fresh_training_and_inference(columns):
    encoder = make_encoder(columns)
    encoder.fit_transform(training_frame(columns))
    replacement = training_frame(columns, replacement=True)
    original = replacement.copy(deep=True)
    actual_x, actual_y = encoder.fit_transform(replacement)

    fresh = make_encoder(columns)
    expected_x, expected_y = fresh.fit_transform(replacement)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
    query = replacement.iloc[[2, 0]].copy()
    query["value"] = np.nan
    actual_x, actual_y = encoder.transform(query)
    expected_x, expected_y = fresh.transform(query)
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)
    pd.testing.assert_frame_equal(replacement, original)


def test_numencoder_single_fit_keeps_feature_order_and_values():
    frame = training_frame(["category"])
    encoder = make_encoder(["category"])
    actual_x, actual_y = encoder.fit_transform(frame)
    expected_x = [
        [1.0, 0.0, 0.0, 1, 0],
        [2.0, 0.0, 0.0, 0, 1],
        [3.0, 0.0, 0.0, 1, 1],
        [4.0, 0.0, 1 / 3, 1, 0],
    ]
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, [[0], [1], [0], [1]])


def test_numencoder_numeric_only_refit():
    encoder = make_encoder([])
    for frame in [training_frame([]), training_frame([], replacement=True)]:
        actual_x, actual_y = encoder.fit_transform(frame)
        np.testing.assert_array_equal(
            actual_x[:, 0], frame["value"].fillna(frame["value"].mean())
        )
        np.testing.assert_array_equal(actual_y[:, 0], frame["label"])
