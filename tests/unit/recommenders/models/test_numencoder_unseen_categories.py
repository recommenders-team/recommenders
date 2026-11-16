# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from recommenders.models.lightgbm.lightgbm_utils import NumEncoder


def fitted_encoder(known_count=3):
    categories = [f"category-{i}" for i in range(known_count)]
    frame = pd.DataFrame(
        {
            "category": categories * 2,
            "value": np.arange(known_count * 2, dtype=float),
            "label": [0, 1] * known_count,
        }
    )
    encoder = NumEncoder(["category"], ["value"], "label", threshold=0, thresrate=1)
    train_x, _ = encoder.fit_transform(frame)
    return encoder, frame, train_x.shape[1]


@pytest.mark.parametrize("known_count", [1, 3, 7])
def test_unseen_categories_keep_distinct_zero_binary_code(known_count):
    encoder, frame, width = fitted_encoder(known_count)
    query = frame.iloc[:known_count].copy()
    known_x, _ = encoder.transform(query)
    query = pd.concat(
        [query, pd.DataFrame({"category": ["unseen"], "value": [1.0], "label": [0]})],
        ignore_index=True,
    )
    original = query.copy(deep=True)
    actual, labels = encoder.transform(query)

    assert actual.shape == (known_count + 1, width)
    np.testing.assert_array_equal(actual[:-1], known_x)
    np.testing.assert_array_equal(actual[-1, 1:], 0)
    assert np.all(np.any(actual[:-1, 3:] != actual[-1, 3:], axis=1))
    np.testing.assert_array_equal(labels[:, 0], query["label"])
    pd.testing.assert_frame_equal(query, original)


def test_entirely_unseen_batch_preserves_fitted_width():
    encoder, _, width = fitted_encoder()
    query = pd.DataFrame(
        {"category": ["unseen-a", "unseen-b"], "value": [4.0, 5.0], "label": [1, 0]}
    )
    actual, labels = encoder.transform(query)
    assert actual.shape == (2, width)
    np.testing.assert_array_equal(actual[:, 0], [4.0, 5.0])
    np.testing.assert_array_equal(actual[:, 1:], 0)
    np.testing.assert_array_equal(labels[:, 0], [1, 0])


def test_unseen_category_does_not_change_other_field_encoding():
    frame = pd.DataFrame(
        {
            "genre": ["a", "b", "a", "b"],
            "region": ["x", "y", "x", "y"],
            "value": [1.0] * 4,
            "label": [0, 1, 1, 0],
        }
    )
    encoder = NumEncoder(
        ["genre", "region"], ["value"], "label", threshold=0, thresrate=1
    )
    encoder.fit_transform(frame)
    expected, _ = encoder.transform(frame.iloc[:1])
    query = frame.iloc[:1].copy()
    query["genre"] = "unseen"
    actual, _ = encoder.transform(query)

    np.testing.assert_array_equal(
        actual[:, [0, 3, 4, 7, 8]], expected[:, [0, 3, 4, 7, 8]]
    )
    np.testing.assert_array_equal(actual[:, [1, 2, 5, 6]], 0)


def test_unseen_categories_keep_existing_rare_bucket():
    frame = pd.DataFrame(
        {
            "category": ["popular"] * 4 + ["rare"],
            "value": [1.0] * 5,
            "label": [0, 1, 0, 1, 1],
        }
    )
    encoder = NumEncoder(["category"], ["value"], "label", threshold=1, thresrate=1)
    encoder.fit_transform(frame)
    expected, _ = encoder.transform(frame.iloc[-1:])
    query = frame.iloc[-1:].copy()
    query["category"] = "unseen"
    actual, _ = encoder.transform(query)
    np.testing.assert_array_equal(actual, expected)
    assert np.any(actual[:, 3:] != 0)


@pytest.mark.parametrize("known_count", [1, 3, 7])
def test_known_category_bits_keep_fitted_ordinal_codes(known_count):
    encoder, frame, _ = fitted_encoder(known_count)
    actual, _ = encoder.transform(frame.iloc[:known_count])
    width = known_count.bit_length()
    expected = [
        [(code >> bit) & 1 for bit in range(width)]
        for code in range(1, known_count + 1)
    ]
    np.testing.assert_array_equal(actual[:, 3:], expected)
