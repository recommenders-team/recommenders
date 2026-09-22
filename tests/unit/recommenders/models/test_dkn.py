# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn

    from recommenders.models.deeprec.models.pytorch.dkn import (
        DKN,
        KCNN,
        _group_by_key,
        glorot_truncated_normal_,
        init_weight_,
    )
except ImportError:
    pass  # skip if torch is not installed


# match the files written by the synthetic_dkn fixture in conftest.py
NEWS_COUNT = 30
DOC_SIZE = 5
DIM = 8
ENTITY_DIM = 6
N_TEST = 20

HISTORY_SIZE = 4
NUM_FILTERS = 3
FILTER_SIZES = [1, 2]
NEWS_DIM = NUM_FILTERS * len(FILTER_SIZES)


@pytest.fixture
def build_model(synthetic_dkn):
    """Build a small seeded DKN, on CPU whatever the machine offers."""

    def build(**overrides):
        kwargs = dict(
            news_feature_file=synthetic_dkn["news"],
            user_history_file=synthetic_dkn["history"],
            word_embedding_file=synthetic_dkn["word"],
            entity_embedding_file=synthetic_dkn["entity"],
            context_embedding_file=synthetic_dkn["context"],
            history_size=HISTORY_SIZE,
            filter_sizes=FILTER_SIZES,
            num_filters=NUM_FILTERS,
            attention_layer_size=4,
            layer_sizes=[6],
            enable_BN=False,
            seed=42,
        )
        kwargs.update(overrides)
        return DKN(**kwargs).to("cpu")

    return build


@pytest.fixture
def first_batch():
    """Read the first mini-batch of a file, as numpy arrays and as tensors."""

    def read(model, path, batch_size):
        np_batch = next(model.iterator.load_data_from_file(path, batch_size))[0]
        return np_batch, model._to_tensors(np_batch)

    return read


@pytest.fixture
def build_kcnn():
    """Build a KCNN on random embeddings of 10 words and 7 entities."""

    def build(use_entity=True, use_context=True, **overrides):
        rng = np.random.RandomState(0)
        kwargs = dict(
            word_embeddings=rng.normal(size=(10, DIM)).astype(np.float32),
            entity_embeddings=(
                rng.normal(size=(7, ENTITY_DIM)).astype(np.float32)
                if use_entity
                else None
            ),
            context_embeddings=(
                rng.normal(size=(7, ENTITY_DIM)).astype(np.float32)
                if use_context
                else None
            ),
            filter_sizes=FILTER_SIZES,
            num_filters=NUM_FILTERS,
            init_method="uniform",
            init_value=0.1,
        )
        kwargs.update(overrides)
        return KCNN(**kwargs)

    return build


def training_loss(model, path):
    """Mean data loss over a file, in eval mode."""
    model.eval()
    with torch.no_grad():
        losses = [
            model._data_loss(model._to_tensors(np_batch)).item()
            for np_batch, _ in model.iterator.load_data_from_file(path, 8)
        ]
    return float(np.mean(losses))


# --------------------------- initializers ---------------------------


@pytest.mark.parametrize(
    "shape, fan_avg",
    [((200, 300), 250.0), ((5000,), 5000.0)],
)
def test_xavier_normal_keeps_the_glorot_variance(shape, fan_avg):
    weight = torch.empty(shape)
    init_weight_(weight, "xavier_normal", init_value=0.1)

    # The truncation at two standard deviations is compensated, so the drawn
    # values keep the Glorot variance 1 / fan_avg; a bias counts its size as both
    # fans.
    assert weight.std().item() == pytest.approx(np.sqrt(1.0 / fan_avg), rel=0.05)
    bound = 2 * np.sqrt(1.0 / fan_avg) / 0.87962566103423978
    assert weight.abs().max().item() <= bound


def test_glorot_truncated_normal_takes_the_fan_average():
    weight = torch.empty(100, 400)
    glorot_truncated_normal_(weight, fan_avg=1000.0)

    assert weight.std().item() == pytest.approx(np.sqrt(1.0 / 1000.0), rel=0.05)


def test_uniform_init_stays_within_the_init_value():
    weight = torch.empty(50, 50)
    init_weight_(weight, "uniform", init_value=0.3)

    assert weight.abs().max().item() <= 0.3
    assert weight.abs().max().item() > 0.25


def test_init_weight_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="uniform or xavier_normal"):
        init_weight_(torch.empty(3), "tnormal", init_value=0.1)


# --------------------------- KCNN ---------------------------


@pytest.mark.parametrize(
    "use_entity, use_context, channels",
    [(False, False, DIM), (True, False, 2 * DIM), (True, True, 3 * DIM)],
)
def test_kcnn_stacks_one_channel_block_per_embedding(
    build_kcnn, use_entity, use_context, channels
):
    kcnn = build_kcnn(use_entity=use_entity, use_context=use_context)
    words = torch.randint(0, 10, (4, DOC_SIZE))
    entities = torch.randint(0, 7, (4, DOC_SIZE))

    assert [conv.in_channels for conv in kcnn.convs] == [channels, channels]
    assert kcnn(words, entities).shape == (4, NEWS_DIM)
    assert kcnn.output_dim == NEWS_DIM


def test_kcnn_matches_the_closed_form(build_kcnn):
    kcnn = build_kcnn(num_filters=1)
    with torch.no_grad():
        for conv in kcnn.convs:
            conv.weight.fill_(1.0)
            conv.bias.zero_()
    words = torch.randint(0, 10, (3, DOC_SIZE))
    entities = torch.randint(0, 7, (3, DOC_SIZE))

    with torch.no_grad():
        out = kcnn(words, entities).numpy()
        tables = [table.numpy() for table in kcnn.embedding_tables()]

    # All-ones filters sum every channel of a window: per position the word,
    # entity and context embeddings, then ReLU and the maximum over positions.
    per_position = (
        tables[0][words.numpy()].sum(-1)
        + tables[1][entities.numpy()].sum(-1)
        + tables[2][entities.numpy()].sum(-1)
    )
    size_1 = np.maximum(per_position, 0).max(axis=1)
    size_2 = np.maximum(per_position[:, :-1] + per_position[:, 1:], 0).max(axis=1)

    assert np.allclose(out, np.stack([size_1, size_2], axis=1), atol=1e-5)


def test_kcnn_maps_entities_and_contexts_through_a_tanh_layer(build_kcnn):
    kcnn = build_kcnn()

    with torch.no_grad():
        word_table, entity_table, context_table = kcnn.embedding_tables()
        expected_entity = torch.tanh(
            kcnn.entity_embedding @ kcnn.entity_transform.weight.T
            + kcnn.entity_transform.bias
        )
        expected_context = torch.tanh(
            kcnn.context_embedding @ kcnn.context_transform.weight.T
            + kcnn.context_transform.bias
        )

    assert word_table is kcnn.word_embedding
    assert entity_table.shape == (7, DIM)
    assert torch.allclose(entity_table, expected_entity)
    assert torch.allclose(context_table, expected_context)


def test_kcnn_fine_tunes_words_but_keeps_entities_and_contexts_fixed(build_kcnn):
    kcnn = build_kcnn()
    parameters = {name for name, _ in kcnn.named_parameters()}
    buffers = {name for name, _ in kcnn.named_buffers()}

    assert "word_embedding" in parameters
    assert buffers == {"entity_embedding", "context_embedding"}


def test_kcnn_copies_the_word_embeddings(build_kcnn):
    words = np.ones((10, DIM), dtype=np.float32)
    kcnn = build_kcnn(word_embeddings=words)

    with torch.no_grad():
        kcnn.word_embedding.add_(1.0)

    assert np.all(words == 1.0)


def test_kcnn_rejects_contexts_without_entities(build_kcnn):
    with pytest.raises(ValueError, match="context_embeddings requires"):
        build_kcnn(use_entity=False, use_context=True)


# --------------------------- DKN components ---------------------------


def test_forward_shape(build_model, first_batch, synthetic_dkn):
    model = build_model()
    _, batch = first_batch(model, synthetic_dkn["train"], 8)

    with torch.no_grad():
        assert model(batch).shape == (8, 1)


def test_forward_scores_the_user_vector_then_the_candidate(
    build_model, first_batch, synthetic_dkn
):
    model = build_model().eval()
    _, batch = first_batch(model, synthetic_dkn["train"], 8)

    with torch.no_grad():
        candidate = model.kcnn(batch["candidate_words"], batch["candidate_entities"])
        clicked = model.kcnn(
            batch["clicked_words"].reshape(-1, DOC_SIZE),
            batch["clicked_entities"].reshape(-1, DOC_SIZE),
        ).view(8, HISTORY_SIZE, NEWS_DIM)
        user = model._attend(clicked, candidate)
        expected = model.scorer(torch.cat([user, candidate], dim=1))

        assert torch.allclose(model(batch), expected, atol=1e-6)


def test_attention_returns_the_article_when_all_clicks_are_equal(build_model):
    model = build_model().eval()
    article = torch.randn(3, 1, NEWS_DIM)

    with torch.no_grad():
        user = model._attend(
            article.expand(3, HISTORY_SIZE, NEWS_DIM), torch.randn(3, NEWS_DIM)
        )

    # the attention weights sum to 1 over the history
    assert torch.allclose(user, article.squeeze(1), atol=1e-6)


def test_attention_averages_every_history_slot_padding_included(build_model):
    model = build_model().eval()
    with torch.no_grad():
        for linear in (*model.attention.linears, model.attention.out):
            linear.weight.zero_()
            linear.bias.zero_()
    clicked = torch.randn(2, HISTORY_SIZE, NEWS_DIM)

    with torch.no_grad():
        user = model._attend(clicked, torch.randn(2, NEWS_DIM))

    # equal attention logits give every slot, padded or not, weight 1 / HISTORY_SIZE
    assert torch.allclose(user, clicked.mean(dim=1), atol=1e-6)


def test_attention_uses_relu_and_the_scorer_sigmoid(build_model):
    model = build_model()

    assert isinstance(model.attention.activation, nn.ReLU)
    assert isinstance(model.scorer.activation, nn.Sigmoid)
    assert [linear.out_features for linear in model.attention.linears] == [4]
    assert [linear.out_features for linear in model.scorer.linears] == [6]
    assert model.attention.linears[0].in_features == 2 * NEWS_DIM
    assert model.scorer.linears[0].in_features == 2 * NEWS_DIM


@pytest.mark.parametrize(
    "enable_BN, expected", [(True, "BatchNorm1d"), (False, "Identity")]
)
def test_inserts_batch_norm_only_when_enabled(build_model, enable_BN, expected):
    model = build_model(enable_BN=enable_BN)

    assert type(model.attention.bns[0]).__name__ == expected
    assert type(model.scorer.bns[0]).__name__ == expected


def test_biases_follow_the_init_method(build_model):
    model = build_model(init_value=0.2)
    biases = [
        linear.bias
        for net in (model.attention, model.scorer)
        for linear in (*net.linears, net.out)
    ] + [conv.bias for conv in model.kcnn.convs]

    for bias in biases:
        assert bias.abs().max().item() <= 0.2
        assert bias.abs().max().item() > 0.0


def test_layer_params_leave_out_batch_norm_and_knowledge_transforms(build_model):
    model = build_model(enable_BN=True)
    expected = [*model.kcnn.convs.parameters()] + [
        param
        for net in (model.attention, model.scorer)
        for linear in (*net.linears, net.out)
        for param in linear.parameters()
    ]

    assert {id(p) for p in model.layer_params} == {id(p) for p in expected}
    assert len(model.layer_params) == len(expected)


def test_rejects_an_unknown_init_method(build_model):
    with pytest.raises(ValueError, match="uniform or xavier_normal"):
        build_model(init_method="tnormal")


# --------------------------- losses ---------------------------


def test_data_loss_is_the_log_loss(build_model, first_batch, synthetic_dkn):
    model = build_model().eval()
    np_batch, batch = first_batch(model, synthetic_dkn["train"], 8)

    with torch.no_grad():
        loss = model._data_loss(batch).item()
        pred = torch.sigmoid(model(batch)).numpy().reshape(-1)

    labels = np_batch["labels"].reshape(-1)
    expected = np.mean(
        -labels * np.log(pred + 1e-7) - (1 - labels) * np.log(1 - pred + 1e-7)
    )
    assert loss == pytest.approx(expected, rel=1e-5)


def test_regular_loss_matches_the_closed_form(build_model):
    model = build_model()

    with torch.no_grad():
        reg = model._regular_loss(
            embed_l2=0.1, embed_l1=0.2, layer_l2=0.3, layer_l1=0.4
        ).item()
        tables = [table.numpy() for table in model.kcnn.embedding_tables()]
    params = [param.detach().numpy() for param in model.layer_params]

    # the L2 terms carry a factor 0.5; the transformed entity and context tables
    # count as embeddings
    expected = sum(0.1 * 0.5 * (t**2).sum() + 0.2 * np.abs(t).sum() for t in tables)
    expected += sum(0.3 * 0.5 * (p**2).sum() + 0.4 * np.abs(p).sum() for p in params)
    assert reg == pytest.approx(expected, rel=1e-5)


# --------------------------- evaluation helpers ---------------------------


def test_groups_labels_and_predictions_by_impression():
    labels, preds = _group_by_key(
        [1, 0, 0, 1], [0.9, 0.2, 0.4, 0.7], ["imp1", "imp2", "imp1", "imp2"]
    )

    assert labels == [[1, 0], [0, 1]]
    assert preds == [[0.9, 0.4], [0.2, 0.7]]


# --------------------------- training lifecycle ---------------------------


def test_fit_decreases_the_training_loss(build_model, synthetic_dkn):
    model = build_model()
    before = training_loss(model, synthetic_dkn["train"])

    model.fit(
        synthetic_dkn["train"],
        synthetic_dkn["valid"],
        epochs=5,
        batch_size=8,
        learning_rate=0.01,
        show_step=100,
    )

    assert training_loss(model, synthetic_dkn["train"]) < before


def test_fit_and_eval_smoke(build_model, synthetic_dkn, tmp_path):
    model = build_model(enable_BN=True)
    model_dir = os.path.join(tmp_path, "model")

    returned = model.fit(
        synthetic_dkn["train"],
        synthetic_dkn["valid"],
        test_file=synthetic_dkn["test"],
        epochs=2,
        batch_size=8,
        embed_l1=0.001,
        layer_l1=0.001,
        max_grad_norm=0.5,
        show_step=1,
        model_dir=model_dir,
        save_epoch=1,
    )

    assert returned is model
    assert sorted(os.listdir(model_dir)) == ["epoch_1", "epoch_2"]
    assert set(model.run_eval(synthetic_dkn["test"], batch_size=8)) == {
        "auc",
        "group_auc",
        "mean_mrr",
        "ndcg@5",
        "ndcg@10",
    }


def test_run_eval_reports_the_requested_metrics(build_model, synthetic_dkn):
    model = build_model()

    res = model.run_eval(
        synthetic_dkn["test"],
        batch_size=8,
        metrics=["auc", "logloss"],
        pairwise_metrics=["mean_mrr"],
    )

    assert set(res) == {"auc", "logloss", "mean_mrr"}


def test_predict_writes_one_probability_per_instance(
    build_model, synthetic_dkn, tmp_path
):
    model = build_model()
    output_file = os.path.join(tmp_path, "output.txt")

    assert model.predict(synthetic_dkn["test"], output_file, batch_size=8) is model
    with open(output_file) as f:
        scores = f.read().strip().split("\n")

    assert len(scores) == N_TEST
    assert all(0.0 <= float(score) <= 1.0 for score in scores)


def test_run_get_embedding_writes_the_kcnn_embedding_of_every_article(
    build_model, synthetic_dkn, tmp_path
):
    model = build_model()
    output_file = os.path.join(tmp_path, "embedding.txt")

    assert (
        model.run_get_embedding(synthetic_dkn["news"], output_file, batch_size=8)
        is model
    )
    with open(output_file) as f:
        lines = [line.split(" ") for line in f.read().strip().split("\n")]

    assert len(lines) == NEWS_COUNT
    news_id, values = lines[0]
    row = model.iterator.news_rows[news_id]
    with torch.no_grad():
        expected = model.kcnn(
            torch.as_tensor(model.iterator.words[row : row + 1]),
            torch.as_tensor(model.iterator.entities[row : row + 1]),
        ).numpy()[0]
    assert np.allclose(np.array(values.split(","), dtype=np.float32), expected)


def test_save_and_load_model_round_trip(
    build_model, first_batch, synthetic_dkn, tmp_path
):
    model = build_model(enable_BN=True).eval()
    _, batch = first_batch(model, synthetic_dkn["test"], 8)
    with torch.no_grad():
        expected = model(batch)

    checkpoint = os.path.join(tmp_path, "epoch_1")
    torch.save(model.state_dict(), checkpoint)

    restored = build_model(enable_BN=True, seed=7).eval()
    assert restored.load_model(checkpoint) is restored
    with torch.no_grad():
        assert torch.allclose(restored(batch), expected, atol=1e-6)


@pytest.mark.gpu
def test_trains_on_the_gpu(build_model, synthetic_dkn):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    model = build_model().to("cuda")

    model.fit(
        synthetic_dkn["train"],
        synthetic_dkn["valid"],
        epochs=1,
        batch_size=8,
        show_step=100,
    )

    assert model.device.type == "cuda"
    assert "auc" in model.run_eval(synthetic_dkn["test"], batch_size=8)
