# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

try:
    import torch

    from recommenders.models.deeprec.io.ffm_dataset import FFMDataset
    from recommenders.models.deeprec.models.pytorch.xdeepfm import CIN, XDeepFMModel
except ImportError:
    pass  # skip if torch is not installed


FEATURE_COUNT = 200
FIELD_COUNT = 6
DIM = 4
N_TRAIN, N_VALID, N_TEST = 40, 20, 20


def _line(label, rng):
    """One FFM line: every field carries exactly one feature, with 1-based indices."""
    parts = [str(label)]
    for field in range(1, FIELD_COUNT + 1):
        feature = field * 20 + rng.randint(1, 11)
        parts.append("{0}:{1}:1".format(field, feature))
    return " ".join(parts)


@pytest.fixture(scope="module")
def synthetic_ffm(tmp_path_factory):
    """Tiny synthetic FFM files, so the test needs no download."""
    d = tmp_path_factory.mktemp("xdeepfm_pt")
    rng = np.random.RandomState(0)
    paths = {}
    for name, n_lines in [("train", N_TRAIN), ("valid", N_VALID), ("test", N_TEST)]:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            for i in range(n_lines):
                f.write(_line(i % 2, rng) + "\n")
        paths[name] = path
    return paths


def _build_model(**overrides):
    kwargs = dict(
        feature_count=FEATURE_COUNT,
        field_count=FIELD_COUNT,
        dim=DIM,
        use_cin_part=True,
        cross_layer_sizes=[4],
        layer_sizes=[8, 8],
        dropout=[0.0, 0.0],
        init_value=0.1,
        seed=42,
    )
    kwargs.update(overrides)
    # Keep the assertions on CPU regardless of the machine the tests run on.
    return XDeepFMModel(**kwargs).to("cpu")


def _first_batch(model, path, batch_size):
    np_batch = next(model.iterator.load_data_from_file(path, batch_size))[0]
    return np_batch, model._to_tensors(np_batch)


def _field_embeddings(model, np_batch):
    """Reference [B, FIELD_COUNT, DIM] field embeddings.

    The synthetic data carries exactly one feature per field, so every bag is a
    single entry and the sum pooling is a plain lookup.
    """
    embedding = model.embedding.detach().numpy()
    ids, values = np_batch["feat_ids"], np_batch["feat_values"]
    return (values[:, None] * embedding[ids]).reshape(-1, FIELD_COUNT, DIM)


# --------------------------- FFMDataset ---------------------------


def test_ffm_dataset_parses_one_line():
    dataset = FFMDataset(FIELD_COUNT)
    label, features, impression_id = dataset.parser_one_line("1 1:5:0.5 3:9:2%imp7")

    assert label == 1.0
    # field and feature indices are 1-based in the file, 0-based in memory
    assert features == [[0, 4, 0.5], [2, 8, 2.0]]
    assert impression_id == "imp7"


def test_ffm_dataset_parses_line_without_impression_id():
    dataset = FFMDataset(FIELD_COUNT)
    _, _, impression_id = dataset.parser_one_line("0 1:5:1")

    assert impression_id == 0


@pytest.mark.parametrize(
    "batch_size, expected_sizes", [(20, [20, 20]), (30, [30, 10]), (40, [40])]
)
def test_ffm_dataset_batches_the_file(synthetic_ffm, batch_size, expected_sizes):
    dataset = FFMDataset(FIELD_COUNT)
    batches = list(dataset.load_data_from_file(synthetic_ffm["train"], batch_size))

    assert [len(ids) for _, ids in batches] == expected_sizes
    for (np_batch, impression_ids), size in zip(batches, expected_sizes):
        assert np_batch["labels"].shape == (size, 1)
        assert np_batch["dnn_offsets"].shape == (size * FIELD_COUNT,)
        assert np_batch["feat_ids"].shape == np_batch["feat_values"].shape


def test_ffm_dataset_groups_features_into_one_bag_per_field(synthetic_ffm):
    dataset = FFMDataset(FIELD_COUNT)
    np_batch, _ = next(
        dataset.load_data_from_file(synthetic_ffm["train"], batch_size=4)
    )

    # Every field of the synthetic data holds exactly one feature, so the bags are
    # consecutive single entries.
    assert np.array_equal(
        np_batch["dnn_offsets"], np.arange(4 * FIELD_COUNT, dtype=np.int64)
    )
    assert np_batch["feat_ids"].shape == (4 * FIELD_COUNT,)


def test_ffm_dataset_sorts_features_by_field(tmp_path):
    path = os.path.join(tmp_path, "shuffled")
    with open(path, "w") as f:
        # fields deliberately out of order, and field 2 carries two features
        f.write("1 3:31:1 1:11:1 2:21:2 2:22:3\n")

    dataset = FFMDataset(field_count=4)
    np_batch, _ = next(dataset.load_data_from_file(path, batch_size=1))

    assert np.array_equal(np_batch["feat_ids"], np.array([10, 20, 21, 30]))
    assert np.array_equal(np_batch["feat_values"], np.array([1.0, 2.0, 3.0, 1.0]))
    # field 0 -> 1 entry, field 1 -> 2 entries, field 2 -> 1 entry, field 3 -> empty
    assert np.array_equal(np_batch["dnn_offsets"], np.array([0, 1, 3, 4]))


def test_ffm_dataset_rejects_field_index_beyond_field_count(tmp_path):
    path = os.path.join(tmp_path, "bad_field")
    with open(path, "w") as f:
        f.write("1 1:11:1 9:91:1\n")

    dataset = FFMDataset(field_count=4)
    with pytest.raises(ValueError, match="field index"):
        next(dataset.load_data_from_file(path, batch_size=1))


# --------------------------- CIN ---------------------------


def test_cin_output_shape():
    cin = CIN(FIELD_COUNT, [8, 4], False, 0.1)
    out = cin(torch.randn(5, FIELD_COUNT, DIM))

    assert out.shape == (5, 1)


def test_cin_first_layer_drops_self_interactions():
    cin = CIN(FIELD_COUNT, [1], False, 0.1)
    with torch.no_grad():
        cin.filters[0].fill_(1.0)
        cin.w_out.fill_(1.0)
        cin.b_out.zero_()

    x = torch.randn(3, FIELD_COUNT, DIM)
    out = cin(x)

    # Only the strictly upper triangular field pairs survive, doubled; the residual
    # connection then adds the summed result to the w_out projection of itself.
    pairwise = torch.einsum("bfd,bgd->bfgd", x, x)
    mask = torch.triu(torch.ones(FIELD_COUNT, FIELD_COUNT), diagonal=1)
    expected = 2 * (pairwise * mask[None, :, :, None]).sum(dim=(1, 2, 3)) * 2

    assert torch.allclose(out.squeeze(-1), expected, atol=1e-5)


def test_cin_rejects_odd_size_on_a_split_layer():
    with pytest.raises(ValueError, match="must be even"):
        CIN(FIELD_COUNT, [5, 4], False, 0.1)


# --------------------------- components ---------------------------


def test_linear_part_matches_closed_form(synthetic_ffm):
    model = _build_model(use_cin_part=False, use_linear_part=True)
    np_batch, batch = _first_batch(model, synthetic_ffm["test"], 8)

    with torch.no_grad():
        logit = model(batch).numpy().reshape(-1)

    weight = model.linear_w.detach().numpy().reshape(-1)
    bias = model.linear_b.item()
    offsets = np.append(
        np_batch["dnn_offsets"][::FIELD_COUNT][1:], len(np_batch["feat_ids"])
    )
    expected, start = [], 0
    for end in offsets:
        entries = slice(start, end)
        expected.append(
            float(
                np.sum(
                    np_batch["feat_values"][entries]
                    * weight[np_batch["feat_ids"][entries]]
                )
            )
            + bias
        )
        start = end

    assert np.allclose(logit, expected, atol=1e-5)


def test_fm_part_matches_closed_form(synthetic_ffm):
    model = _build_model(use_cin_part=False, use_fm_part=True)
    np_batch, batch = _first_batch(model, synthetic_ffm["test"], 8)

    with torch.no_grad():
        logit = model(batch).numpy().reshape(-1)

    # Each instance's field bags concatenate into its full feature set.
    field_embed = _field_embeddings(model, np_batch)
    embedding = model.embedding.detach().numpy()
    ids = np_batch["feat_ids"].reshape(-1, FIELD_COUNT)
    values = np_batch["feat_values"].reshape(-1, FIELD_COUNT)
    summed = field_embed.sum(axis=1)
    squared = ((values[:, :, None] * embedding[ids]) ** 2).sum(axis=1)
    expected = 0.5 * (summed**2 - squared).sum(axis=1)

    assert np.allclose(logit, expected, atol=1e-5)


def test_dnn_part_reads_the_sum_pooled_field_embeddings(synthetic_ffm):
    model = _build_model(use_cin_part=False, use_dnn_part=True)
    np_batch, batch = _first_batch(model, synthetic_ffm["test"], 8)

    with torch.no_grad():
        logit = model(batch)
        field_embed = torch.as_tensor(
            _field_embeddings(model, np_batch), dtype=torch.float32
        )
        expected = model.dnn(field_embed.reshape(-1, FIELD_COUNT * DIM))

    assert torch.allclose(logit, expected, atol=1e-4)


def test_enabled_components_sum_their_logits(synthetic_ffm):
    """The full model's logit is the sum of each component's own logit."""
    parts = {
        "use_linear_part": True,
        "use_fm_part": True,
        "use_cin_part": True,
        "use_dnn_part": True,
    }
    full = _build_model(**parts)
    np_batch, batch = _first_batch(full, synthetic_ffm["test"], 8)

    total = torch.zeros(8, 1)
    for part in parts:
        single = _build_model(**{p: (p == part) for p in parts})
        single.load_state_dict(
            {k: v for k, v in full.state_dict().items() if k in single.state_dict()}
        )
        with torch.no_grad():
            total = total + single(single._to_tensors(np_batch))

    with torch.no_grad():
        assert torch.allclose(full(batch), total, atol=1e-4)


def test_forward_shape_for_a_mixed_component_pair(synthetic_ffm):
    parts = {"use_cin_part": True, "use_dnn_part": True}
    model = _build_model(**parts)
    _, batch = _first_batch(model, synthetic_ffm["test"], 8)

    with torch.no_grad():
        assert model(batch).shape == (8, 1)


def test_regression_method_returns_the_raw_logit(synthetic_ffm):
    model = _build_model(method="regression")
    _, batch = _first_batch(model, synthetic_ffm["test"], 8)

    with torch.no_grad():
        logit = model(batch)
        assert torch.equal(model._get_pred(logit), logit)


# --------------------------- construction guards ---------------------------


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="regression or classification"):
        _build_model(method="ranking")


def test_requires_at_least_one_component():
    with pytest.raises(ValueError, match="must be enabled"):
        _build_model(use_cin_part=False)


# --------------------------- training lifecycle ---------------------------


def test_fit_and_eval_smoke(synthetic_ffm):
    model = _build_model(use_linear_part=True, use_dnn_part=True)

    returned = model.fit(
        synthetic_ffm["train"],
        synthetic_ffm["valid"],
        epochs=2,
        batch_size=8,
        learning_rate=0.01,
        loss="log_loss",
        embed_l2=0.001,
        layer_l2=0.001,
        cross_l2=0.001,
        show_step=100,
    )

    assert returned is model
    assert set(model.run_eval(synthetic_ffm["test"], batch_size=8)) == {
        "auc",
        "logloss",
    }


@pytest.mark.parametrize("loss", ["cross_entropy_loss", "log_loss", "square_loss"])
def test_fit_runs_with_every_loss(synthetic_ffm, loss):
    model = _build_model()
    model.fit(
        synthetic_ffm["train"],
        synthetic_ffm["valid"],
        epochs=1,
        batch_size=8,
        loss=loss,
        show_step=100,
    )


def test_fit_rejects_an_unknown_loss(synthetic_ffm):
    model = _build_model()
    with pytest.raises(ValueError, match="this loss not defined"):
        model.fit(
            synthetic_ffm["train"],
            synthetic_ffm["valid"],
            epochs=1,
            batch_size=8,
            loss="hinge",
            show_step=100,
        )


def test_predict_writes_one_score_per_instance(synthetic_ffm, tmp_path):
    model = _build_model()
    output_file = os.path.join(tmp_path, "output.txt")

    assert model.predict(synthetic_ffm["test"], output_file, batch_size=8) is model
    with open(output_file) as f:
        scores = f.read().strip().split("\n")

    assert len(scores) == N_TEST
    assert all(0.0 <= float(score) <= 1.0 for score in scores)


def test_save_and_load_model_round_trip(synthetic_ffm, tmp_path):
    model = _build_model(use_linear_part=True, use_dnn_part=True)
    _, batch = _first_batch(model, synthetic_ffm["test"], 8)
    with torch.no_grad():
        expected = model(batch)

    checkpoint = os.path.join(tmp_path, "epoch_1")
    torch.save(model.state_dict(), checkpoint)

    restored = _build_model(use_linear_part=True, use_dnn_part=True, seed=7)
    assert restored.load_model(checkpoint) is restored
    with torch.no_grad():
        assert torch.allclose(restored(batch), expected, atol=1e-6)
