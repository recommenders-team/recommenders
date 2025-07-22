# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import tempfile
import numpy as np
import pandas as pd
import torch

from recommenders.models.embdotbias.data_loader import RecoDataLoader, RecoDataset
from recommenders.models.embdotbias.model import EmbeddingDotBias
from recommenders.models.embdotbias.training_utils import Trainer, predict_rating
from recommenders.utils.constants import (
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
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


@pytest.fixture(scope="module")
def sample_model_params():
    """Sample model parameters for testing."""
    return {"n_factors": 50, "n_users": 6, "n_items": 11, "y_range": (1.0, 5.0)}


@pytest.fixture(scope="module")
def sample_classes():
    """Create sample classes mapping for testing."""
    return {
        DEFAULT_USER_COL: ["1", "2", "3", "4", "5"],
        DEFAULT_ITEM_COL: ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    }


@pytest.mark.gpu
def test_embedding_dot_bias_weight_method(sample_classes):
    """Test EmbeddingDotBias weight method."""
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=sample_classes, y_range=(1.0, 5.0)
    )

    # Test user weight
    user_weight = model.weight(["1", "2"], is_item=False)
    assert user_weight.shape == (2, 10)

    # Test item weight
    item_weight = model.weight(["1", "2"], is_item=True)
    assert item_weight.shape == (2, 10)


@pytest.mark.gpu
def test_embedding_dot_bias_from_classes(sample_classes):
    """Test EmbeddingDotBias.from_classes method."""
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=sample_classes, y_range=(1.0, 5.0)
    )

    assert model.classes == sample_classes
    assert model.user == DEFAULT_USER_COL
    assert model.item == DEFAULT_ITEM_COL
    assert model.u_weight.num_embeddings == len(sample_classes[DEFAULT_USER_COL])
    assert model.i_weight.num_embeddings == len(sample_classes[DEFAULT_ITEM_COL])


@pytest.mark.gpu
def test_embedding_dot_bias_init(sample_model_params):
    """Test EmbeddingDotBias initialization."""
    model = EmbeddingDotBias(**sample_model_params)

    assert model.u_weight.num_embeddings == sample_model_params["n_users"]
    assert model.i_weight.num_embeddings == sample_model_params["n_items"]
    assert model.u_bias.num_embeddings == sample_model_params["n_users"]
    assert model.i_bias.num_embeddings == sample_model_params["n_items"]
    assert model.y_range == sample_model_params["y_range"]


@pytest.mark.gpu
def test_embedding_dot_bias_forward(sample_model_params):
    """Test EmbeddingDotBias forward pass."""
    model = EmbeddingDotBias(**sample_model_params)

    # Create sample input
    batch_size = 3
    x = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)

    output = model(x)

    assert output.shape == (batch_size,)
    assert output.dtype == torch.float


@pytest.mark.gpu
def test_trainer_init(sample_model_params):
    """Test Trainer initialization."""
    model = EmbeddingDotBias(**sample_model_params)
    trainer = Trainer(model, learning_rate=0.001, weight_decay=0.01)

    assert trainer.model == model
    assert trainer.optimizer is not None
    assert trainer.loss_fn is not None
    assert trainer.device is not None


@pytest.mark.gpu
def test_trainer_train_epoch(sample_ratings_data):
    """Test Trainer train_epoch method."""
    # Create dataloader
    dl = RecoDataLoader.from_df(
        sample_ratings_data, valid_pct=0.2, batch_size=4, seed=42
    )

    # Create model
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=dl.classes, y_range=(1.0, 5.0)
    )

    trainer = Trainer(model, learning_rate=0.001)

    # Train for one epoch
    loss = trainer.train_epoch(dl.train)

    assert isinstance(loss, float)
    assert loss >= 0


@pytest.mark.gpu
def test_trainer_fit(sample_ratings_data):
    """Test Trainer fit method."""
    # Create dataloader
    dl = RecoDataLoader.from_df(
        sample_ratings_data, valid_pct=0.2, batch_size=4, seed=42
    )

    # Create model
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=dl.classes, y_range=(1.0, 5.0)
    )

    trainer = Trainer(model, learning_rate=0.001)

    # Fit for 2 epochs
    trainer.fit(dl.train, dl.valid, n_epochs=2)

    # Model should be trained
    assert model.training is False  # Should be in eval mode after training


@pytest.mark.gpu
def test_trainer_validate(sample_ratings_data):
    """Test Trainer validate method."""
    # Create dataloader
    dl = RecoDataLoader.from_df(
        sample_ratings_data, valid_pct=0.2, batch_size=4, seed=42
    )

    # Create model
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=dl.classes, y_range=(1.0, 5.0)
    )

    trainer = Trainer(model, learning_rate=0.001)

    # Validate
    loss = trainer.validate(dl.valid)

    assert isinstance(loss, float)
    assert loss >= 0


@pytest.mark.gpu
def test_predict_rating(sample_classes):
    """Test predict_rating function."""
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=sample_classes, y_range=(1.0, 5.0)
    )

    prediction = predict_rating(model, "1", "1")

    assert isinstance(prediction, float)
    assert 1.0 <= prediction <= 5.0  # Should be within y_range


@pytest.mark.gpu
def test_full_pipeline(sample_ratings_data):
    """Test the full pipeline from data loading to training."""

    # Create dataloader
    dl = RecoDataLoader.from_df(
        sample_ratings_data, valid_pct=0.2, batch_size=4, seed=42
    )

    # Create model
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=dl.classes, y_range=(1.0, 5.0)
    )

    # Create trainer
    trainer = Trainer(model, learning_rate=0.001)

    # Train for one epoch
    train_loss = trainer.train_epoch(dl.train)
    valid_loss = trainer.validate(dl.valid)

    # Make prediction
    prediction = predict_rating(model, "1", "1")

    assert isinstance(train_loss, float)
    assert isinstance(valid_loss, float)
    assert isinstance(prediction, float)
    assert train_loss >= 0
    assert valid_loss >= 0
    assert 1.0 <= prediction <= 5.0


@pytest.mark.gpu
@pytest.mark.parametrize(
    "entity_ids,is_item,expected_exception",
    [
        (["999"], True, KeyError),  # Non-existent item
        (["999"], False, KeyError),  # Non-existent user
        ([], True, None),  # Empty list
        ([], False, None),  # Empty list
    ],
)
def test_get_idx_edge_cases(sample_classes, entity_ids, is_item, expected_exception):
    model = EmbeddingDotBias.from_classes(
        n_factors=10, classes=sample_classes, y_range=(1.0, 5.0)
    )
    if expected_exception:
        with pytest.raises(expected_exception):
            model._get_idx(entity_ids, is_item=is_item)
    else:
        result = model._get_idx(entity_ids, is_item=is_item)
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 0


def test_reco_dataset(sample_ratings_data):
    """Test RecoDataset `__len__` and `__getitem__`."""
    users = sample_ratings_data[DEFAULT_USER_COL].values
    items = sample_ratings_data[DEFAULT_ITEM_COL].values
    ratings = sample_ratings_data[DEFAULT_RATING_COL].values

    dataset = RecoDataset(users, items, ratings)

    assert len(dataset) == len(ratings)

    user_item_tensor, rating_tensor = dataset[0]
    assert user_item_tensor.shape == (2,)
    assert rating_tensor.shape == (1,)
    assert user_item_tensor[0] == users[0]
    assert user_item_tensor[1] == items[0]
    assert rating_tensor[0] == ratings[0]


@pytest.mark.gpu
def test_reco_dataset(sample_ratings_data):
    """Test RecoDataset `__len__` and `__getitem__`."""
    users = sample_ratings_data[DEFAULT_USER_COL].values
    items = sample_ratings_data[DEFAULT_ITEM_COL].values
    ratings = sample_ratings_data[DEFAULT_RATING_COL].values

    dataset = RecoDataset(users, items, ratings)

    assert len(dataset) == len(ratings)

    user_item_tensor, rating_tensor = dataset[0]
    assert user_item_tensor.shape == (2,)
    assert rating_tensor.shape == (1,)
    assert user_item_tensor[0] == users[0]
    assert user_item_tensor[1] == items[0]
    assert rating_tensor[0] == ratings[0]


@pytest.mark.gpu
def test_model_serialization(sample_model_params):
    """Test saving and loading of EmbeddingDotBias model."""

    model = EmbeddingDotBias(**sample_model_params)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/model.pt"
        torch.save(model.state_dict(), path)
        loaded_model = EmbeddingDotBias(**sample_model_params)
        loaded_model.load_state_dict(torch.load(path))
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)
