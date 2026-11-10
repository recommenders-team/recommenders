# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pickle

import numpy as np
import pytest

try:
    import torch

    from recommenders.models.deeprec.models.sequential.pytorch.caser import (
        CaserConv,
        CaserModel,
    )
except ImportError:
    pass  # skip if torch is not installed


@pytest.fixture(scope="module")
def synthetic_caser(tmp_path_factory):
    """Tiny synthetic vocabs + train/valid/test files (no Amazon download)."""
    d = tmp_path_factory.mktemp("caser_pt")
    n_items = 40
    # index 0 is reserved for OOV/PAD, so vocab length exceeds the max real index
    userdict = {f"U{i}": i for i in range(0, 11)}
    itemdict = {f"I{i}": i for i in range(0, n_items + 1)}
    catedict = {f"C{i}": i for i in range(0, 6)}
    paths = {}
    for name, dct in [
        ("user_vocab.pkl", userdict),
        ("item_vocab.pkl", itemdict),
        ("category_vocab.pkl", catedict),
    ]:
        p = os.path.join(d, name)
        with open(p, "wb") as f:
            pickle.dump(dct, f)
        paths[name] = p

    rng = np.random.RandomState(0)

    def hist(n):
        items = ",".join(f"I{rng.randint(1, n_items + 1)}" for _ in range(n))
        cates = ",".join(f"C{rng.randint(1, 6)}" for _ in range(n))
        base = 1300000000
        times = ",".join(str(base + k * 86400 * 3) for k in range(n))
        return items, cates, times

    def line(label, item, cate, hlen):
        u = f"U{rng.randint(1, 11)}"
        items, cates, times = hist(hlen)
        t = 1300000000 + hlen * 86400 * 5
        return f"{label}\t{u}\t{item}\t{cate}\t{t}\t{items}\t{cates}\t{times}"

    # train: positives only (need_sample generates negatives in-batch)
    with open(os.path.join(d, "train_data"), "w") as f:
        for _ in range(60):
            f.write(
                line(1, f"I{rng.randint(1, n_items + 1)}", "C1", rng.randint(2, 8))
                + "\n"
            )

    # valid/test: groups of (1 pos + 4 neg) consecutive lines
    def grouped(path, n_groups, num_ngs):
        with open(path, "w") as f:
            for _ in range(n_groups):
                pos = f"I{rng.randint(1, n_items + 1)}"
                hlen = rng.randint(2, 8)
                f.write(line(1, pos, "C1", hlen) + "\n")
                for _ in range(num_ngs):
                    f.write(
                        line(0, f"I{rng.randint(1, n_items + 1)}", "C1", hlen) + "\n"
                    )

    grouped(os.path.join(d, "valid_data"), 8, 4)
    grouped(os.path.join(d, "test_data"), 8, 4)

    return {
        "dir": str(d),
        "user_vocab": paths["user_vocab.pkl"],
        "item_vocab": paths["item_vocab.pkl"],
        "cate_vocab": paths["category_vocab.pkl"],
    }


def _build_model(paths, **overrides):
    kwargs = dict(
        user_vocab=paths["user_vocab"],
        item_vocab=paths["item_vocab"],
        cate_vocab=paths["cate_vocab"],
        max_seq_length=10,
        min_seq_length=1,
        L=3,
        n_v=4,
        n_h=5,
        layer_sizes=[16, 8],
        activation=["relu", "relu"],
        dropout=[0.0, 0.0],
        seed=42,
    )
    kwargs.update(overrides)
    return CaserModel(**kwargs)


def _first_batch(paths, model, filename="test_data"):
    return next(
        b
        for b in model.iterator.load_data_from_file(
            os.path.join(paths["dir"], filename),
            batch_size=100,
            min_seq_length=model.min_seq_length,
            batch_num_ngs=0,
        )
        if b
    )


# --------------------------- CaserConv ---------------------------


def test_caser_conv_matches_tf_conv1d_reference():
    """Compare with the TF ``conv1d`` formulas (kernel layout ``[k, in, out]``).

    TF: ``y[b, t, f] = relu(sum_{i, c} x[b, t + i, c] * K[i, c, f] + bias[f])`` with
    ``valid`` padding; the vertical branch convolves the transposed ``[B, E, T]``
    input with a kernel spanning all ``E`` positions, and the horizontal branches
    max-pool over time.
    """
    torch.manual_seed(0)
    b, t, e, n_v, n_h, L = 3, 7, 6, 4, 5, 3
    conv = CaserConv(t, e, n_v, n_h, L)
    # biases are zero at init; randomise them so that they are checked as well
    for p in conv.parameters():
        if p.dim() == 1:
            torch.nn.init.normal_(p)
    x = torch.randn(b, t, e)

    k_v = conv.vertical.weight.permute(2, 1, 0)  # [E, T, n_v]
    out_v = torch.relu(
        torch.einsum("bec,ecf->bf", x.transpose(1, 2), k_v) + conv.vertical.bias
    )
    expected = [out_v]
    for h, layer in enumerate(conv.horizontal, start=1):
        k_h = layer.weight.permute(2, 1, 0)  # [h, E, n_h]
        windows = torch.stack([x[:, i : t - h + 1 + i, :] for i in range(h)], dim=2)
        y = torch.relu(torch.einsum("btic,icf->btf", windows, k_h) + layer.bias)
        assert y.shape == (b, t - h + 1, n_h)
        expected.append(y.max(dim=1).values)
    expected = torch.cat(expected, dim=1)

    out = conv(x)
    assert out.shape == (b, n_v + L * n_h)
    assert torch.allclose(out, expected, atol=1e-5)


def test_caser_conv_init_is_glorot_uniform_with_zero_bias():
    torch.manual_seed(0)
    t, e, n_v, n_h, L = 50, 32, 128, 128, 3
    conv = CaserConv(t, e, n_v, n_h, L)
    for layer in [conv.vertical, *conv.horizontal]:
        fan_in = layer.in_channels * layer.kernel_size[0]
        fan_out = layer.out_channels * layer.kernel_size[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        assert layer.weight.abs().max().item() <= limit + 1e-6
        assert layer.weight.abs().max().item() > 0.9 * limit
        assert torch.count_nonzero(layer.bias) == 0


# --------------------------- Model ---------------------------


@pytest.mark.parametrize("L", [0, 11])
def test_caser_rejects_invalid_L(synthetic_caser, L):
    with pytest.raises(ValueError):
        _build_model(synthetic_caser, L=L)


def test_caser_model_output_dim(synthetic_caser):
    model = _build_model(synthetic_caser)
    # (n_v + L * n_h) for items and for categories, plus the target embedding
    assert model.logit_fcn.linears[0].in_features == 2 * (4 + 3 * 5) + 32 + 8


def test_caser_forward_shapes(synthetic_caser):
    model = _build_model(synthetic_caser)
    np_batch = _first_batch(synthetic_caser, model)
    batch = model._to_tensors(np_batch)
    model.eval()
    with torch.no_grad():
        logit = model.forward(batch)
    assert logit.shape == (np_batch["labels"].shape[0], 1)
    assert torch.isfinite(logit).all()


def test_caser_min_seq_length_is_used(synthetic_caser):
    model = _build_model(synthetic_caser, min_seq_length=5)
    assert model.min_seq_length == 5
    np_batch = _first_batch(synthetic_caser, model)
    assert (np_batch["mask"].sum(axis=1) >= 5).all()


def test_caser_fit_eval_and_load_model(synthetic_caser, tmp_path):
    model = _build_model(synthetic_caser)
    d = synthetic_caser["dir"]
    model.fit(
        os.path.join(d, "train_data"),
        os.path.join(d, "valid_data"),
        epochs=1,
        batch_size=100,
        train_num_ngs=4,
        valid_num_ngs=4,
        embed_l2=1e-4,
        layer_l2=1e-4,
        save_model=True,
        model_dir=str(tmp_path),
        early_stop=0,
    )
    test_file = os.path.join(d, "test_data")
    res = model.run_eval(test_file, num_ngs=4, batch_size=100)
    for key in [
        "auc",
        "logloss",
        "mean_mrr",
        "ndcg@2",
        "ndcg@4",
        "ndcg@6",
        "group_auc",
    ]:
        assert key in res

    # the best checkpoint can be restored into a fresh model
    restored = _build_model(synthetic_caser, seed=0).load_model(str(tmp_path))
    res_restored = restored.run_eval(test_file, num_ngs=4, batch_size=100)
    assert res_restored["auc"] == pytest.approx(res["auc"])

    out_file = str(tmp_path / "pred.txt")
    model.predict(test_file, out_file, batch_size=100)
    with open(out_file) as f:
        assert len(f.read().split()) == 8 * 5
