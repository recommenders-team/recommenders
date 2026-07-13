# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pickle

import numpy as np
import pytest

try:
    import torch

    from recommenders.models.deeprec.deeprec_utils import prepare_hparams
    from recommenders.models.deeprec.io.sequential_dataset_pytorch import (
        SequentialDataset,
    )
    from recommenders.models.deeprec.models.sequential.pytorch.rnn_cell_pytorch import (
        Time4LSTMCell,
        time4lstm_scan,
    )
    from recommenders.models.deeprec.models.sequential.pytorch.sequential_base_pytorch import (
        Attention,
        FcnNet,
    )
    from recommenders.models.deeprec.models.sequential.pytorch.sli_rec_pytorch import (
        AttentionFcn,
        SLI_RECModel,
    )
except ImportError:
    pass  # skip if torch is not installed


CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "recommenders",
    "models",
    "deeprec",
    "config",
    "sli_rec.yaml",
)


@pytest.fixture(scope="module")
def synthetic_slirec(tmp_path_factory):
    """Tiny synthetic vocabs + train/valid/test files (no Amazon download)."""
    d = tmp_path_factory.mktemp("slirec_pt")
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


@pytest.fixture(scope="module")
def hparams(synthetic_slirec):
    return prepare_hparams(
        CONFIG_PATH,
        embed_l2=0.0,
        layer_l2=0.0,
        learning_rate=0.001,
        epochs=1,
        batch_size=100,
        show_step=1000,
        MODEL_DIR=os.path.join(synthetic_slirec["dir"], "model"),
        user_vocab=synthetic_slirec["user_vocab"],
        item_vocab=synthetic_slirec["item_vocab"],
        cate_vocab=synthetic_slirec["cate_vocab"],
        need_sample=True,
        train_num_ngs=4,
    )


# --------------------------- Time4LSTMCell ---------------------------


def test_time4lstm_cell_matches_manual_equations():
    torch.manual_seed(0)
    b, d, h = 3, 5, 4
    cell = Time4LSTMCell(d, h)
    inputs = torch.randn(b, d + 2)
    c_prev, m_prev = torch.randn(b, h), torch.randn(b, h)

    m, (c, m2) = cell(inputs, (c_prev, m_prev))

    # manual reference of the gate equations
    x = inputs[:, :d]
    t_last, t_now = inputs[:, d : d + 1], inputs[:, d + 1 : d + 2]
    tni = torch.tanh(t_now * cell.time_input_w1 + cell.time_input_bias1)
    tli = torch.tanh(t_last * cell.time_input_w2 + cell.time_input_bias2)
    tns = x @ cell.time_kernel_w1 + tni @ cell.time_kernel_t1 + cell.time_bias1
    tls = x @ cell.time_kernel_w2 + tli @ cell.time_kernel_t2 + cell.time_bias2
    i, j, f, o = torch.split(
        torch.cat([x, m_prev], 1) @ cell.W_lstm + cell.b_lstm, h, dim=1
    )
    o = o + tni @ cell.o_kernel_t1 + tli @ cell.o_kernel_t2
    c_ref = torch.sigmoid(f + 1.0) * torch.sigmoid(tls) * c_prev + torch.sigmoid(
        i
    ) * torch.sigmoid(tns) * torch.tanh(j)
    m_ref = torch.sigmoid(o) * torch.tanh(c_ref)

    assert torch.allclose(m, m_ref, atol=1e-6)
    assert torch.allclose(c, c_ref, atol=1e-6)
    assert torch.allclose(m, m2)


def test_time4lstm_cell_time_columns_not_symmetric():
    """time_to_now (col -1) and time_from_first_action (col -2) must differ in effect."""
    torch.manual_seed(1)
    b, d, h = 2, 4, 3
    cell = Time4LSTMCell(d, h)
    inp = torch.randn(b, d + 2)
    state = (torch.zeros(b, h), torch.zeros(b, h))
    swapped = inp.clone()
    swapped[:, -1], swapped[:, -2] = inp[:, -2], inp[:, -1]
    m1, _ = cell(inp, state)
    m2, _ = cell(swapped, state)
    assert not torch.allclose(m1, m2)


def test_time4lstm_scan_zeros_padded_steps():
    torch.manual_seed(2)
    b, t, d, h = 3, 6, 4, 5
    cell = Time4LSTMCell(d, h)
    inputs = torch.randn(b, t, d + 2)
    seq_len = torch.tensor([t, t - 2, 3])
    out = time4lstm_scan(cell, inputs, seq_len)
    assert out.shape == (b, t, h)
    for row in range(b):
        pad = out[row, seq_len[row] :]
        if pad.numel():
            assert torch.count_nonzero(pad) == 0


# --------------------------- Attention blocks ---------------------------


def test_asvd_attention_weights_sum_to_one_over_time():
    torch.manual_seed(3)
    b, t, e = 2, 5, 8
    att = Attention(e, e, "tnormal", 0.01)
    inputs = torch.randn(b, t, e)
    # recompute weights to assert they form a softmax over T (incl. padding)
    ai = torch.einsum("bte,ef->btf", inputs, att.attention_mat)
    logits = torch.einsum("btf,f->bt", ai, att.query)
    w = torch.softmax(logits, dim=-1)
    assert torch.allclose(w.sum(dim=1), torch.ones(b), atol=1e-5)
    out = att(inputs)
    assert out.shape == (b, t, e)


def test_attention_fcn_masks_padded_positions():
    torch.manual_seed(4)
    b, t, h, e = 2, 5, 6, 6
    afcn = AttentionFcn(
        h, e, [8, 4], ["relu", "relu"], [0.0, 0.0], True, False, "tnormal", 0.01
    )
    afcn.eval()
    query = torch.randn(b, e)
    user_emb = torch.randn(b, t, h)
    mask = torch.ones(b, t)
    mask[:, 3:] = 0.0  # last two steps padded
    out = afcn(query, user_emb, mask)
    # weights on padded positions must be ~0, so those rows contribute ~0
    assert out.shape == (b, t, h)
    assert torch.allclose(out[:, 3:], torch.zeros(b, t - 3, h), atol=1e-5)


def test_fcn_net_output_shape_2d_and_3d():
    net = FcnNet(12, [8, 4], ["relu", "relu"], [0.0, 0.0], True, True, "tnormal", 0.01)
    net.eval()
    assert net(torch.randn(7, 12)).shape == (7, 1)
    assert net(torch.randn(7, 5, 12)).shape == (7, 5, 1)


# --------------------------- Dataset ---------------------------


def test_dataset_time_features_floor_and_log(synthetic_slirec):
    hp = prepare_hparams(
        CONFIG_PATH,
        user_vocab=synthetic_slirec["user_vocab"],
        item_vocab=synthetic_slirec["item_vocab"],
        cate_vocab=synthetic_slirec["cate_vocab"],
    )
    ds = SequentialDataset(hp)
    # two history stamps 1 day apart, current 2 days after last
    line = "1\tU1\tI1\tC1\t1300172800\tI2,I3\tC1,C1\t1300000000,1300086400"
    parsed = ds.parser_one_line(line)
    _, _, _, _, item_hist, cate_hist, cur, tdiff, tffa, ttn = parsed
    assert item_hist == [2, 3]
    # time_to_now: (cur - t)/86400 floored at 0.5 then log
    expected_ttn = np.log(
        [max((cur - 1300000000) / 86400, 0.5), max((cur - 1300086400) / 86400, 0.5)]
    )
    assert np.allclose(ttn, expected_ttn)
    assert len(tdiff) == len(tffa) == len(ttn) == 2


def test_dataset_eval_batch_shapes_and_mask(hparams):
    ds = SequentialDataset(hparams)
    batch = next(
        b
        for b in ds.load_data_from_file(
            os.path.join(os.path.dirname(hparams.user_vocab), "test_data"),
            batch_num_ngs=0,
        )
        if b
    )
    n = batch["labels"].shape[0]
    assert batch["item_history"].shape == (n, hparams.max_seq_length)
    assert batch["mask"].shape == (n, hparams.max_seq_length)
    # mask row sum equals number of valid history steps (>=1)
    assert (batch["mask"].sum(axis=1) >= 1).all()


def test_dataset_train_negative_sampling_pattern(hparams):
    ds = SequentialDataset(hparams)
    batch = next(
        b
        for b in ds.load_data_from_file(
            os.path.join(os.path.dirname(hparams.user_vocab), "train_data"),
            batch_num_ngs=4,
        )
        if b
    )
    labels = batch["labels"].reshape(-1, 5)
    assert (labels[:, 0] == 1).all()  # positive first
    assert (labels[:, 1:] == 0).all()  # then negatives
    # history is duplicated across the group
    ih = batch["item_history"].reshape(-1, 5, hparams.max_seq_length)
    assert (ih[:, 0:1] == ih).all()


# --------------------------- Model ---------------------------


def test_slirec_dimension_coupling_asserted(synthetic_slirec):
    bad = prepare_hparams(
        CONFIG_PATH,
        hidden_size=64,  # != item(32)+cate(8)=40
        user_vocab=synthetic_slirec["user_vocab"],
        item_vocab=synthetic_slirec["item_vocab"],
        cate_vocab=synthetic_slirec["cate_vocab"],
        train_num_ngs=4,
    )
    with pytest.raises(ValueError):
        SLI_RECModel(bad, SequentialDataset, seed=42)


def test_slirec_forward_shapes(hparams):
    model = SLI_RECModel(hparams, SequentialDataset, seed=42)
    ds = model.iterator
    np_batch = next(
        b
        for b in ds.load_data_from_file(
            os.path.join(os.path.dirname(hparams.user_vocab), "test_data"),
            batch_num_ngs=0,
        )
        if b
    )
    batch = model._to_tensors(np_batch)
    with torch.no_grad():
        logit = model.forward(batch)
    n = np_batch["labels"].shape[0]
    assert logit.shape == (n, 1)


def test_slirec_fit_and_eval_smoke(hparams):
    model = SLI_RECModel(hparams, SequentialDataset, seed=42)
    d = os.path.dirname(hparams.user_vocab)
    model.fit(
        os.path.join(d, "train_data"), os.path.join(d, "valid_data"), valid_num_ngs=4
    )
    res = model.run_eval(os.path.join(d, "test_data"), num_ngs=4)
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
