# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pandas as pd
import pytest

try:
    import torch
    from recommenders.models.two_tower.data_loader import TwoTowerData
    from recommenders.models.two_tower.model import Tower, TwoTowerModel
    from recommenders.models.two_tower.training_utils import (
        Trainer,
        in_batch_softmax_loss,
    )
    from recommenders.models.two_tower.utils import recommend_k_items
except ImportError:
    pass  # skip this import if we are in cpu environment

from recommenders.utils.constants import DEFAULT_ITEM_COL, DEFAULT_USER_COL


@pytest.fixture(scope="module")
def clustered_interactions():
    """Two groups of users that each interact with a different half of the items.

    Users 0-9 interact with items 0-9 and users 10-19 with items 10-19. Each user
    leaves out a few items of their own group, which a model that learned the
    structure should recommend before any item of the other group.
    """
    rng = np.random.default_rng(42)
    rows = []
    for user in range(20):
        group_items = np.arange(10) if user < 10 else np.arange(10, 20)
        for item in rng.choice(group_items, size=6, replace=False):
            rows.append((f"u{user}", f"i{item}"))
    return pd.DataFrame(rows, columns=[DEFAULT_USER_COL, DEFAULT_ITEM_COL])


@pytest.mark.gpu
def test_tower_without_features_is_embedding_lookup():
    tower = Tower(n_entities=5, embedding_dim=8, hidden_units=(), output_dim=16)
    assert isinstance(tower.mlp, torch.nn.Identity)
    assert tower.output_dim == 8
    out = tower(torch.tensor([0, 1, 4]))
    assert out.shape == (3, 8)
    assert torch.allclose(out.norm(dim=1), torch.ones(3), atol=1e-5)


@pytest.mark.gpu
def test_tower_with_features():
    features = np.random.rand(5, 3).astype(np.float32)
    tower = Tower(
        n_entities=5,
        embedding_dim=4,
        hidden_units=(8,),
        output_dim=6,
        features=features,
        normalize=False,
    )
    assert tower.features.shape == (5, 3)
    out = tower(torch.tensor([0, 2]))
    assert out.shape == (2, 6)
    # The features are part of the state dict, but not trainable
    assert "features" in tower.state_dict()
    assert all(p.data_ptr() != tower.features.data_ptr() for p in tower.parameters())


@pytest.mark.gpu
def test_tower_invalid_arguments():
    with pytest.raises(ValueError):
        Tower(n_entities=0)
    with pytest.raises(ValueError):
        Tower(n_entities=5, features=np.zeros((4, 3)))


@pytest.mark.gpu
def test_two_tower_model_shapes():
    model = TwoTowerModel(
        n_users=7,
        n_items=11,
        embedding_dim=8,
        hidden_units=(16,),
        output_dim=4,
        item_features=np.random.rand(11, 3),
    )
    user_idx = torch.tensor([0, 3, 6])
    item_idx = torch.tensor([1, 5, 10])
    assert model.encode_users(user_idx).shape == (3, 4)
    assert model.encode_items(item_idx).shape == (3, 4)
    assert model.encode_items().shape == (11, 4)
    scores = model(user_idx, item_idx)
    assert scores.shape == (3,)
    # Normalized vectors give cosine similarities
    assert torch.all(scores.abs() <= 1 + 1e-5)


@pytest.mark.gpu
def test_two_tower_model_mismatched_towers():
    # A user tower without MLP outputs embedding_dim=8, while the item tower
    # with features outputs output_dim=4
    with pytest.raises(ValueError):
        TwoTowerModel(
            n_users=3,
            n_items=3,
            embedding_dim=8,
            hidden_units=(),
            output_dim=4,
            item_features=np.zeros((3, 2)),
        )


@pytest.mark.gpu
def test_two_tower_model_save_load(tmp_path):
    features = np.random.rand(6, 2)
    model = TwoTowerModel(n_users=4, n_items=6, item_features=features)
    path = os.path.join(tmp_path, "two_tower.pth")
    torch.save(model.state_dict(), path)

    loaded = TwoTowerModel(n_users=4, n_items=6, item_features=np.zeros((6, 2)))
    loaded.load_state_dict(torch.load(path))
    model.eval()
    loaded.eval()
    assert torch.allclose(model.encode_items(), loaded.encode_items())
    assert torch.allclose(loaded.item_tower.features, torch.tensor(features).float())


@pytest.mark.gpu
def test_in_batch_softmax_loss_prefers_aligned_vectors():
    torch.manual_seed(0)
    vectors = torch.nn.functional.normalize(torch.randn(8, 16), dim=1)
    item_idx = torch.arange(8)
    aligned = in_batch_softmax_loss(vectors, vectors, item_idx, temperature=0.1)
    shuffled = in_batch_softmax_loss(
        vectors, vectors[torch.randperm(8)], item_idx, temperature=0.1
    )
    assert aligned < shuffled
    with pytest.raises(ValueError):
        in_batch_softmax_loss(vectors, vectors, item_idx, temperature=0)


@pytest.mark.gpu
def test_in_batch_softmax_loss_masks_accidental_hits():
    user_vectors = torch.eye(3)
    # Rows 0 and 1 share the same positive item, so neither should be a
    # negative for the other
    item_idx = torch.tensor([5, 5, 7])
    item_vectors = torch.tensor([[1.0, 0, 0], [1.0, 0, 0], [0, 0, 1.0]])
    loss = in_batch_softmax_loss(user_vectors, item_vectors, item_idx, temperature=1.0)

    logits = user_vectors @ item_vectors.T
    expected = torch.stack(
        [
            -torch.log_softmax(logits[0, [0, 2]], dim=0)[0],
            -torch.log_softmax(logits[1, [1, 2]], dim=0)[0],
            -torch.log_softmax(logits[2], dim=0)[2],
        ]
    ).mean()
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


@pytest.mark.gpu
def test_in_batch_softmax_loss_logq_correction():
    user_vectors = torch.eye(2)
    item_vectors = torch.eye(2)
    item_idx = torch.tensor([0, 1])
    no_correction = in_batch_softmax_loss(
        user_vectors, item_vectors, item_idx, temperature=1.0
    )
    log_q = torch.log(torch.tensor([0.5, 0.5]))
    uniform = in_batch_softmax_loss(
        user_vectors, item_vectors, item_idx, temperature=1.0, item_log_q=log_q
    )
    # A uniform correction shifts every logit by the same amount
    assert uniform.item() == pytest.approx(no_correction.item(), rel=1e-6)
    skewed = in_batch_softmax_loss(
        user_vectors,
        item_vectors,
        item_idx,
        temperature=1.0,
        item_log_q=torch.log(torch.tensor([0.9, 0.1])),
    )
    assert skewed.item() != pytest.approx(no_correction.item(), rel=1e-3)


@pytest.mark.gpu
def test_trainer_invalid_item_counts():
    model = TwoTowerModel(n_users=3, n_items=4)
    with pytest.raises(ValueError):
        Trainer(model, item_counts=np.ones(5), device="cpu")


@pytest.mark.gpu
def test_trainer_learns_clusters(clustered_interactions):
    torch.manual_seed(0)
    data = TwoTowerData(clustered_interactions)
    model = TwoTowerModel(
        n_users=data.n_users,
        n_items=data.n_items,
        embedding_dim=16,
        hidden_units=(),
    )
    trainer = Trainer(
        model,
        learning_rate=0.05,
        temperature=0.1,
        item_counts=data.item_counts,
        device="cpu",
    )
    train_dl = data.loader(batch_size=32, seed=0)
    history = trainer.fit(train_dl, n_epochs=30, valid_dl=train_dl, verbose=False)

    assert len(history) == 30
    assert set(history[0]) == {"epoch", "train_loss", "valid_loss"}
    assert history[-1]["train_loss"] < history[0]["train_loss"]

    recs = recommend_k_items(model, data, top_k=2)
    group = recs[DEFAULT_USER_COL].str[1:].astype(int) < 10
    item_group = recs[DEFAULT_ITEM_COL].str[1:].astype(int) < 10
    # Most recommendations should come from the user's own group of items
    assert (group == item_group).mean() > 0.9
