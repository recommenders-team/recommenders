# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pickle

import numpy as np
import pytest

try:
    import torch

    from recommenders.models.newsrec.io.mind_dataset import MINDDataset
    from recommenders.models.newsrec.models.layers import (
        AdditiveAttention,
        SelfAttention,
    )
    from recommenders.models.newsrec.models.lstur import LSTURModel
    from recommenders.models.newsrec.models.naml import NAMLModel
    from recommenders.models.newsrec.models.npa import NPAModel
    from recommenders.models.newsrec.models.nrms import NRMSModel
except ImportError:
    pass  # skip if torch is not installed


WORDS = ["w{0}".format(i) for i in range(1, 20)]
TITLE_SIZE = 5
BODY_SIZE = 6
HIS_SIZE = 4
NPRATIO = 2

# (title, abstract, category, subcategory); N12 has no known title word.
NEWS = [
    ("w1 w2 w3", "w4 w5", "c1", "s1"),
    ("w2 w3 w4 w5 w6 w7 w8", "w1", "c1", "s2"),
    ("w5 zz w6", "", "c2", "s3"),
    ("w7", "w8 w9 w10 w11 w12 w13 w14", "c2", "s3"),
    ("w8 w9", "w1 w2", "c3", "s4"),
    ("w10 w11 w12", "w3", "c3", "s4"),
    ("w13", "w4 w5 w6", "c1", "s1"),
    ("w14 w15", "w7", "c9", "s9"),
    ("w16 w17 w18", "w8", "c2", "s2"),
    ("w19 w1", "w9", "c3", "s1"),
    ("w3 w6 w9", "w10", "c1", "s3"),
    ("zz qq", "w11", "c2", "s4"),
]

# (user, history, impression); U99 is missing from the user dictionary.
BEHAVIORS = [
    ("U1", "", "N1-1 N2-0 N3-0 N4-0"),
    ("U2", "N1", "N5-0 N6-1 N7-0 N8-1"),
    ("U3", "N1 N2 N3 N4 N5 N6", "N7-1 N8-0"),
    ("U4", "N12 N3", "N9-0 N10-0 N11-1 N2-0 N3-0"),
    ("U99", "N4 N5", "N1-0 N12-1 N6-0"),
    ("U5", "N6 N7 N8", "N2-1 N3-1 N4-0 N5-0 N9-0"),
    ("U6", "N9", "N10-0 N11-0 N1-1"),
    ("U1", "N2 N10 N11", "N4-1 N5-0 N6-0 N7-0"),
]


@pytest.fixture(scope="module")
def mind_files(tmp_path_factory):
    """Tiny MIND split and dictionaries, so the tests need no download."""
    d = tmp_path_factory.mktemp("mind")
    rng = np.random.RandomState(0)
    paths = {
        "news": os.path.join(d, "news.tsv"),
        "behaviors": os.path.join(d, "behaviors.tsv"),
    }
    with open(paths["news"], "w") as f:
        for i, (title, abstract, vert, subvert) in enumerate(NEWS, start=1):
            f.write(
                "N{0}\t{1}\t{2}\t{3}\t{4}\turl\t[]\t[]\n".format(
                    i, vert, subvert, title, abstract
                )
            )
    with open(paths["behaviors"], "w") as f:
        for i, (user, history, impression) in enumerate(BEHAVIORS, start=1):
            f.write(
                "{0}\t{1}\t11/11/2019 9:05:58 AM\t{2}\t{3}\n".format(
                    i, user, history, impression
                )
            )
    dicts = {
        "word_dict": {word: i for i, word in enumerate(WORDS, start=1)},
        "uid2index": {"U{0}".format(i): i for i in range(1, 7)},
        "vert_dict": {"c1": 1, "c2": 2, "c3": 3},
        "subvert_dict": {"s1": 1, "s2": 2, "s3": 3, "s4": 4},
    }
    for name, value in dicts.items():
        paths[name] = os.path.join(d, name + ".pkl")
        with open(paths[name], "wb") as f:
            pickle.dump(value, f)
    paths["embedding"] = os.path.join(d, "embedding.npy")
    np.save(paths["embedding"], rng.normal(size=(len(WORDS) + 1, 8)).astype(np.float32))
    return paths


@pytest.fixture
def dataset(mind_files):
    with open(mind_files["word_dict"], "rb") as f:
        word_dict = pickle.load(f)
    with open(mind_files["uid2index"], "rb") as f:
        uid2index = pickle.load(f)
    with open(mind_files["vert_dict"], "rb") as f:
        vert_dict = pickle.load(f)
    with open(mind_files["subvert_dict"], "rb") as f:
        subvert_dict = pickle.load(f)
    return MINDDataset(
        mind_files["news"],
        mind_files["behaviors"],
        word_dict,
        TITLE_SIZE,
        HIS_SIZE,
        uid2index=uid2index,
        body_size=BODY_SIZE,
        vert_dict=vert_dict,
        subvert_dict=subvert_dict,
    )


MODELS = {
    "nrms": lambda files: NRMSModel(
        files["embedding"],
        files["word_dict"],
        title_size=TITLE_SIZE,
        his_size=HIS_SIZE,
        head_num=2,
        head_dim=3,
        attention_hidden_dim=4,
        seed=42,
    ),
    "naml": lambda files: NAMLModel(
        files["embedding"],
        files["word_dict"],
        files["vert_dict"],
        files["subvert_dict"],
        title_size=TITLE_SIZE,
        body_size=BODY_SIZE,
        his_size=HIS_SIZE,
        vert_emb_dim=3,
        subvert_emb_dim=3,
        filter_num=6,
        attention_hidden_dim=4,
        seed=42,
    ),
    "lstur_ini": lambda files: LSTURModel(
        files["embedding"],
        files["word_dict"],
        files["uid2index"],
        title_size=TITLE_SIZE,
        his_size=HIS_SIZE,
        filter_num=6,
        attention_hidden_dim=4,
        seed=42,
    ),
    "lstur_con": lambda files: LSTURModel(
        files["embedding"],
        files["word_dict"],
        files["uid2index"],
        title_size=TITLE_SIZE,
        his_size=HIS_SIZE,
        filter_num=6,
        combine="con",
        attention_hidden_dim=4,
        seed=42,
    ),
    "npa": lambda files: NPAModel(
        files["embedding"],
        files["word_dict"],
        files["uid2index"],
        title_size=TITLE_SIZE,
        his_size=HIS_SIZE,
        user_emb_dim=3,
        filter_num=6,
        attention_hidden_dim=4,
        seed=42,
    ),
}


@pytest.fixture(params=sorted(MODELS))
def model(request, mind_files):
    """Each model, small and seeded, on CPU whatever the machine offers."""
    return MODELS[request.param](mind_files).to("cpu")


def test_dataset_reads_articles_into_rows(dataset):
    assert dataset.news["title"].shape == (len(NEWS) + 1, TITLE_SIZE)
    assert dataset.news["title"][0].tolist() == [0] * TITLE_SIZE
    assert dataset.news["title"][1].tolist() == [1, 2, 3, 0, 0]
    assert dataset.news["title"][2].tolist() == [2, 3, 4, 5, 6]
    assert dataset.news["title"][3].tolist() == [5, 0, 6, 0, 0]
    assert dataset.news["title"][12].tolist() == [0] * TITLE_SIZE
    assert dataset.news["body"][4].tolist() == [8, 9, 10, 11, 12, 13]
    assert dataset.news["vert"].tolist() == [0, 1, 1, 2, 2, 3, 3, 1, 0, 2, 3, 1, 2]
    assert dataset.news["subvert"][8] == 0


def test_dataset_reads_titles_only_without_body_size(mind_files):
    with open(mind_files["word_dict"], "rb") as f:
        word_dict = pickle.load(f)
    dataset = MINDDataset(
        mind_files["news"], mind_files["behaviors"], word_dict, TITLE_SIZE, HIS_SIZE
    )
    assert list(dataset.news) == ["title"]
    assert dataset.users.tolist() == [0] * len(BEHAVIORS)


def test_dataset_keeps_the_last_clicks_left_padded(dataset):
    assert dataset.histories.shape == (len(BEHAVIORS), HIS_SIZE)
    assert dataset.histories[0].tolist() == [0, 0, 0, 0]
    assert dataset.histories[1].tolist() == [0, 0, 0, 1]
    assert dataset.histories[2].tolist() == [3, 4, 5, 6]


def test_dataset_reads_impressions(dataset):
    assert dataset.users.tolist() == [1, 2, 3, 4, 0, 5, 6, 1]
    assert dataset.impressions[1].tolist() == [5, 6, 7, 8]
    assert dataset.labels[1].tolist() == [0, 1, 0, 1]


def test_train_batches_put_the_click_first(dataset):
    batches = list(dataset.load_train_batches(3, NPRATIO, np.random.default_rng(0)))
    candidates = np.concatenate([batch["candidate_rows"] for batch in batches])
    clicks = sum(int(labels.sum()) for labels in dataset.labels)
    assert [len(batch["users"]) for batch in batches] == [3, 3, 3, 1]
    assert candidates.shape == (clicks, NPRATIO + 1)
    for batch in batches:
        for rows, history, user in zip(
            batch["candidate_rows"], batch["clicked_rows"], batch["users"]
        ):
            # A user and a history identify one impression of the fixture.
            i = next(
                i
                for i in range(len(dataset.users))
                if dataset.users[i] == user and (dataset.histories[i] == history).all()
            )
            labels = dict(zip(dataset.impressions[i], dataset.labels[i]))
            negatives = [row for row in rows[1:] if row != 0]
            assert labels[rows[0]] == 1
            assert all(labels[row] == 0 for row in negatives)
            assert len(set(negatives)) == len(negatives)


def test_train_batches_pad_missing_negatives_with_row_0(dataset):
    batch = next(dataset.load_train_batches(100, 3, np.random.default_rng(0)))
    padded = batch["candidate_rows"][batch["candidate_rows"][:, 0] == 7]
    assert padded.tolist() == [[7, 8, 0, 0]]


def test_train_batches_follow_the_generator(dataset):
    def first(seed):
        return next(dataset.load_train_batches(4, NPRATIO, np.random.default_rng(seed)))

    np.testing.assert_array_equal(
        first(1)["candidate_rows"], first(1)["candidate_rows"]
    )


def test_additive_attention_leaves_out_masked_positions():
    torch.manual_seed(0)
    attention = AdditiveAttention(3, 4)
    x = torch.randn(2, 5, 3)
    mask = torch.tensor([[True, True, False, True, False], [False] * 5])
    out = attention(x, mask)
    np.testing.assert_allclose(
        out[0].detach().numpy(),
        attention(x[:1, [0, 1, 3]])[0].detach().numpy(),
        rtol=1e-5,
    )
    assert out[1].abs().sum() == 0


def test_self_attention_matches_the_formula():
    torch.manual_seed(0)
    attention = SelfAttention(3, 2, 4)
    x = torch.randn(2, 5, 3)

    def heads(linear):
        return linear(x).view(2, 5, 2, 4).transpose(1, 2)

    q, k, v = heads(attention.query), heads(attention.key), heads(attention.value)
    expected = torch.softmax(q @ k.transpose(-1, -2) / 2.0, dim=-1) @ v
    np.testing.assert_allclose(
        attention(x).detach().numpy(),
        expected.transpose(1, 2).reshape(2, 5, 8).detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_forward_scores_every_candidate(model, mind_files):
    dataset = model.load_dataset(mind_files["news"], mind_files["behaviors"])
    batch = next(dataset.load_train_batches(3, NPRATIO, np.random.default_rng(0)))
    batch = {name: torch.as_tensor(value) for name, value in batch.items()}
    logits = model.forward(model._news_tensors(dataset), batch)
    assert logits.shape == (3, NPRATIO + 1)


# 5 encodes NPA one impression at a time, 9 two at a time.
@pytest.mark.parametrize("batch_size", [5, 9])
def test_predict_matches_forward(model, mind_files, batch_size):
    scores = model.predict(
        mind_files["news"], mind_files["behaviors"], batch_size=batch_size
    )
    dataset = model.load_dataset(mind_files["news"], mind_files["behaviors"])
    news = model._news_tensors(dataset)
    model.eval()
    assert [len(score) for score in scores] == [len(i) for i in dataset.impressions]
    for i, impression in enumerate(dataset.impressions):
        batch = {
            "candidate_rows": torch.as_tensor(impression).unsqueeze(0),
            "clicked_rows": torch.as_tensor(dataset.histories[i : i + 1]),
            "users": torch.as_tensor(dataset.users[i : i + 1]),
        }
        with torch.no_grad():
            expected = model.forward(news, batch)[0].numpy()
        np.testing.assert_allclose(scores[i], expected, rtol=1e-4, atol=1e-6)


def test_fit_and_run_eval(model, mind_files, capsys):
    files = (mind_files["news"], mind_files["behaviors"])
    before = model.run_eval(*files)
    assert model.fit(*files, *files, epochs=2, batch_size=3, npratio=NPRATIO) is model
    after = model.run_eval(*files, metrics=["group_auc", "mean_mrr"])
    assert set(before) == {"group_auc", "mean_mrr", "ndcg@5", "ndcg@10"}
    assert set(after) == {"group_auc", "mean_mrr"}
    assert "at epoch 2" in capsys.readouterr().out


def test_fit_stops_at_the_step_limit(model, mind_files, capsys):
    files = (mind_files["news"], mind_files["behaviors"])
    model.fit(*files, *files, *files, epochs=1, batch_size=2, step_limit=1)
    out = capsys.readouterr().out
    assert "eval info" in out and "test info" in out


def test_seed_fixes_the_initial_scores(mind_files):
    files = (mind_files["news"], mind_files["behaviors"])
    first, second = (MODELS["nrms"](mind_files).to("cpu") for _ in range(2))
    for a, b in zip(first.predict(*files), second.predict(*files)):
        np.testing.assert_array_equal(a, b)


def test_lstur_skips_padded_history_slots(mind_files):
    model = MODELS["lstur_ini"](mind_files).to("cpu").eval()
    clicked = torch.randn(2, 3, 6)
    padded = torch.cat([torch.zeros(2, 2, 6), clicked], dim=1)
    users = torch.tensor([1, 0])
    np.testing.assert_allclose(
        model.encode_user(padded, users).detach().numpy(),
        model.encode_user(clicked, users).detach().numpy(),
        rtol=1e-6,
    )


def test_lstur_rejects_an_unknown_combine(mind_files):
    with pytest.raises(ValueError, match="combine"):
        LSTURModel(
            mind_files["embedding"],
            mind_files["word_dict"],
            mind_files["uid2index"],
            combine="add",
        )


def test_user_tables_keep_a_row_for_the_last_user(mind_files):
    for name in ("lstur_ini", "npa"):
        model = MODELS[name](mind_files)
        assert model.user_embedding.num_embeddings == 7
    naml = MODELS["naml"](mind_files)
    assert naml.vert_embedding.num_embeddings == 4
    assert naml.subvert_embedding.num_embeddings == 5
