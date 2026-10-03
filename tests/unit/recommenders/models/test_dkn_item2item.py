# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

try:
    import torch

    from recommenders.models.deeprec.models.pytorch.dkn_item2item import (
        DKNItem2Item,
    )
except ImportError:
    pass  # skip if torch is not installed


# match the files written by the synthetic_dkn fixture in conftest.py
NEWS_COUNT = 30
NEG_NUM = 2

NUM_FILTERS = 3
FILTER_SIZES = [1, 2]
NEWS_DIM = NUM_FILTERS * len(FILTER_SIZES)


@pytest.fixture
def build_model(synthetic_dkn):
    """Build a small seeded DKNItem2Item, on CPU whatever the machine offers."""

    def build(**overrides):
        kwargs = dict(
            news_feature_file=synthetic_dkn["news"],
            word_embedding_file=synthetic_dkn["word"],
            neg_num=NEG_NUM,
            entity_embedding_file=synthetic_dkn["entity"],
            context_embedding_file=synthetic_dkn["context"],
            filter_sizes=FILTER_SIZES,
            num_filters=NUM_FILTERS,
            seed=42,
        )
        kwargs.update(overrides)
        return DKNItem2Item(**kwargs).to("cpu")

    return build


@pytest.fixture
def first_batch():
    """Read the first mini-batch of a file as tensors."""

    def read(model, path, batch_size):
        np_batch = next(model.iterator.load_data_from_file(path, batch_size))[0]
        return model._to_tensors(np_batch)

    return read


@pytest.fixture
def self_related_file(synthetic_dkn, tmp_path):
    """Groups whose related target is the source article itself."""
    with open(synthetic_dkn["item2item_valid"]) as f:
        news_ids = f.read().split()
    path = os.path.join(tmp_path, "self_related")
    with open(path, "w") as f:
        for start in range(0, len(news_ids), NEG_NUM + 2):
            group = news_ids[start : start + NEG_NUM + 2]
            f.write("\n".join([group[0], group[0]] + group[2:]) + "\n")
    return path


def training_loss(model, path):
    """Summed data loss over a file, in eval mode."""
    model.eval()
    with torch.no_grad():
        return sum(
            model._data_loss(model._to_tensors(np_batch)).item()
            for np_batch, _ in model.iterator.load_data_from_file(path, 4)
        )


# --------------------------- components ---------------------------


def test_forward_scores_every_target_of_a_group(
    build_model, first_batch, synthetic_dkn
):
    model = build_model()
    batch = first_batch(model, synthetic_dkn["item2item_train"], 4)

    with torch.no_grad():
        assert model(batch).shape == (4, NEG_NUM + 1)


def test_article_embeddings_are_unit_norm(build_model, first_batch, synthetic_dkn):
    model = build_model()
    batch = first_batch(model, synthetic_dkn["item2item_train"], 4)

    with torch.no_grad():
        news = model._embed_news(
            batch["words"].flatten(0, 1), batch["entities"].flatten(0, 1)
        )

    assert news.shape == (4 * (NEG_NUM + 2), NEWS_DIM)
    assert torch.allclose(news.norm(dim=-1), torch.ones(len(news)), atol=1e-5)


def test_a_zero_embedding_stays_zero(build_model, first_batch, synthetic_dkn):
    model = build_model()
    with torch.no_grad():
        model.doc_transform.weight.zero_()
    batch = first_batch(model, synthetic_dkn["item2item_train"], 4)

    with torch.no_grad():
        news = model._embed_news(
            batch["words"].flatten(0, 1), batch["entities"].flatten(0, 1)
        )

    assert torch.equal(news, torch.zeros_like(news))


def test_scores_are_cosine_similarities_to_the_source(
    build_model, first_batch, self_related_file
):
    model = build_model()
    batch = first_batch(model, self_related_file, 5)

    with torch.no_grad():
        scores = model(batch)
        news = model._embed_news(
            batch["words"].flatten(0, 1), batch["entities"].flatten(0, 1)
        ).view(5, NEG_NUM + 2, NEWS_DIM)

    expected = torch.einsum("bd,btd->bt", news[:, 0], news[:, 1:])
    assert torch.allclose(scores, expected, atol=1e-6)
    # the source article is its own related target
    assert torch.allclose(scores[:, 0], torch.ones(5), atol=1e-5)


def test_data_loss_sums_the_negative_log_softmax_of_the_related_target(
    build_model, first_batch, synthetic_dkn
):
    model = build_model()
    batch = first_batch(model, synthetic_dkn["item2item_train"], 4)

    with torch.no_grad():
        loss = model._data_loss(batch).item()
        scores = model(batch).numpy()

    softmax = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
    assert loss == pytest.approx(-np.log(softmax[:, 0] + 1e-10).sum(), rel=1e-5)


def test_layer_params_hold_the_convolutions_and_the_document_transform(
    build_model,
):
    model = build_model()
    expected = [*model.kcnn.convs.parameters(), model.doc_transform.weight]

    assert model.doc_transform.bias is None
    assert {id(p) for p in model.layer_params} == {id(p) for p in expected}
    assert len(model.layer_params) == len(expected)


# --------------------------- training lifecycle ---------------------------


def test_fit_decreases_the_training_loss(build_model, synthetic_dkn):
    model = build_model()
    before = training_loss(model, synthetic_dkn["item2item_train"])

    returned = model.fit(
        synthetic_dkn["item2item_train"],
        synthetic_dkn["item2item_valid"],
        epochs=5,
        batch_size=4,
        learning_rate=0.01,
        max_grad_norm=0.5,
        show_step=100,
    )

    assert returned is model
    assert training_loss(model, synthetic_dkn["item2item_train"]) < before


@pytest.mark.parametrize(
    "metrics, expected",
    [
        (None, {"group_auc", "mean_mrr", "ndcg@5", "ndcg@10"}),
        (["auc"], {"auc", "group_auc", "mean_mrr", "ndcg@5", "ndcg@10"}),
    ],
)
def test_run_eval_reports_pairwise_metrics_per_group(
    build_model, synthetic_dkn, metrics, expected
):
    model = build_model()

    res = model.run_eval(
        synthetic_dkn["item2item_valid"], batch_size=2, metrics=metrics
    )

    assert set(res) == expected


def test_run_eval_takes_the_first_target_as_the_positive(
    build_model, self_related_file
):
    model = build_model()

    res = model.run_eval(self_related_file, batch_size=2)

    # a source is closer to itself than to any other article
    assert res["group_auc"] == 1.0
    assert res["mean_mrr"] == 1.0


def test_run_get_embedding_writes_unit_vectors(build_model, synthetic_dkn, tmp_path):
    model = build_model()
    output_file = os.path.join(tmp_path, "embedding.txt")

    assert (
        model.run_get_embedding(synthetic_dkn["news"], output_file, batch_size=8)
        is model
    )
    with open(output_file) as f:
        lines = [line.split(" ") for line in f.read().strip().split("\n")]

    assert len(lines) == NEWS_COUNT
    vectors = np.array([values.split(",") for _, values in lines], dtype=np.float32)
    assert vectors.shape == (NEWS_COUNT, NEWS_DIM)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_save_and_load_model_round_trip(
    build_model, first_batch, synthetic_dkn, tmp_path
):
    model = build_model()
    batch = first_batch(model, synthetic_dkn["item2item_valid"], 4)
    with torch.no_grad():
        expected = model(batch)

    checkpoint = os.path.join(tmp_path, "epoch_1")
    torch.save(model.state_dict(), checkpoint)

    restored = build_model(seed=7)
    assert restored.load_model(checkpoint) is restored
    with torch.no_grad():
        assert torch.allclose(restored(batch), expected, atol=1e-6)
