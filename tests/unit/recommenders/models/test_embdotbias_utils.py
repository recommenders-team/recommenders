import numpy as np
import pandas as pd
import pytest
import torch

from recommenders.models.embdotbias.utils import cartesian_product, score
from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)


@pytest.fixture(scope="module")
def sample_ratings_data():
    """Create fixed sample ratings data for testing."""
    data = {
        DEFAULT_USER_COL: [1, 4, 8, 5, 7, 10, 3],
        DEFAULT_ITEM_COL: [1, 3, 14, 17, 4, 18, 8],
        DEFAULT_RATING_COL: [
            3.493193,
            2.323592,
            1.254233,
            2.243929,
            2.300733,
            3.918425,
            3.550230,
        ],
    }
    return pd.DataFrame(data)


def test_cartesian_product_two_arrays():
    a = np.array([1, 2])
    b = np.array([3, 4])
    result = cartesian_product(a, b)
    expected = np.array([[1, 3], [1, 4], [2, 3], [2, 4]])
    np.testing.assert_array_equal(result, expected)


def test_cartesian_product_three_arrays():
    a = np.array([1, 2])
    b = np.array([3, 4])
    c = np.array([5, 6])
    result = cartesian_product(a, b, c)
    expected = np.array(
        [
            [1, 3, 5],
            [1, 3, 6],
            [1, 4, 5],
            [1, 4, 6],
            [2, 3, 5],
            [2, 3, 6],
            [2, 4, 5],
            [2, 4, 6],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_cartesian_product_single_array():
    a = np.array([1, 2, 3])
    result = cartesian_product(a)
    expected = np.array([[1], [2], [3]])
    np.testing.assert_array_equal(result, expected)


def test_cartesian_product_empty_array():
    a = np.array([])
    b = np.array([1, 2])
    result = cartesian_product(a, b)
    expected = np.empty((0, 2))
    np.testing.assert_array_equal(result, expected)


def test_score(sample_ratings_data):
    """Test score function."""

    # Create a dummy model
    class DummyModel:
        def __init__(self, classes):
            self.classes = classes

        def _get_idx(self, entity_ids, is_item=True):
            entity_map = (
                self.classes[DEFAULT_USER_COL]
                if not is_item
                else self.classes[DEFAULT_ITEM_COL]
            )
            return torch.tensor(
                [entity_map.index(x) for x in entity_ids if x in entity_map]
            )

        def forward(self, x):
            return torch.ones(x.shape[0])

        def to(self, device):
            return self

    classes = {
        DEFAULT_USER_COL: list(sample_ratings_data[DEFAULT_USER_COL].unique()),
        DEFAULT_ITEM_COL: list(sample_ratings_data[DEFAULT_ITEM_COL].unique()),
    }
    model = DummyModel(classes)

    # Test with top_k
    result = score(model, sample_ratings_data, top_k=2)
    assert isinstance(result, pd.DataFrame)
    assert result.groupby(DEFAULT_USER_COL).size().max() <= 2

    # Test without top_k
    result = score(model, sample_ratings_data)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_ratings_data)

    # Test with unknown users/items
    test_df_new = pd.DataFrame({DEFAULT_USER_COL: [999, 1], DEFAULT_ITEM_COL: [999, 1]})
    # Calling score with mismatched data lengths should raise ValueError
    with pytest.raises(ValueError):
        score(model, test_df_new)
