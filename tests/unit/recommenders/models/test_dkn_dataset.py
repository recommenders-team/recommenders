# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

from recommenders.models.deeprec.io.dkn_dataset import (
    DKNDataset,
    DKNItem2ItemDataset,
    NewsDataset,
)

# match the files written by the synthetic_dkn fixture in conftest.py
DOC_SIZE = 5
NEG_NUM = 2

HISTORY_SIZE = 4


@pytest.fixture
def write_file(tmp_path):
    """Write lines to a file under tmp_path and return its path."""

    def write(name, lines):
        path = os.path.join(tmp_path, name)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        return path

    return write


@pytest.fixture
def small_news(write_file):
    """Three articles of three words, with distinct word and entity indices."""
    return write_file(
        "news",
        [
            "N1 11,12,13 1,0,2",
            "N2 21,22,23 0,3,0",
            "N3 31,32,33 4,4,0",
        ],
    )


# --------------------------- news features ---------------------------


def test_loads_the_news_features_with_a_trailing_padding_row(small_news):
    dataset = NewsDataset(small_news)

    assert dataset.news_rows == {"N1": 0, "N2": 1, "N3": 2}
    assert np.array_equal(
        dataset.words, [[11, 12, 13], [21, 22, 23], [31, 32, 33], [0, 0, 0]]
    )
    assert np.array_equal(
        dataset.entities, [[1, 0, 2], [0, 3, 0], [4, 4, 0], [0, 0, 0]]
    )
    assert dataset.words.dtype == np.int64
    assert dataset.entities.dtype == np.int64


def test_rejects_entities_misaligned_with_words(write_file):
    path = write_file("news", ["N1 11,12,13 1,0", "N2 21,22,23 0,3"])

    with pytest.raises(ValueError, match="one entity index per word"):
        NewsDataset(path)


@pytest.mark.parametrize(
    "batch_size, expected_sizes", [(10, [10, 10, 10]), (20, [20, 10]), (30, [30])]
)
def test_batches_the_articles_to_embed(synthetic_dkn, batch_size, expected_sizes):
    dataset = NewsDataset(synthetic_dkn["news"])
    batches = list(dataset.load_infer_data_from_file(synthetic_dkn["news"], batch_size))

    assert [len(ids) for _, ids in batches] == expected_sizes
    for (np_batch, _), size in zip(batches, expected_sizes):
        assert np_batch["words"].shape == (size, DOC_SIZE)
        assert np_batch["entities"].shape == (size, DOC_SIZE)
        assert np_batch["words"].dtype == np.int64
        assert np_batch["entities"].dtype == np.int64


def test_reads_the_articles_to_embed_from_the_file_itself(small_news, write_file):
    dataset = NewsDataset(small_news)
    # N9 is unknown to the news features: the inference file carries its own.
    infile = write_file("infer", ["N9 91,92,93 9,0,9", "N2 21,22,23 0,3,0"])

    np_batch, news_ids = next(dataset.load_infer_data_from_file(infile, 8))

    assert news_ids == ["N9", "N2"]
    assert np.array_equal(np_batch["words"], [[91, 92, 93], [21, 22, 23]])
    assert np.array_equal(np_batch["entities"], [[9, 0, 9], [0, 3, 0]])


# --------------------------- DKN ---------------------------


def test_keeps_the_latest_clicks_and_pads_the_rest(small_news, write_file):
    history = write_file("history", ["U1 N1,N2,N3", "U2 N3", "U3"])
    dataset = DKNDataset(small_news, history, history_size=2)

    padding_row = 3
    assert np.array_equal(dataset.user_history["U1"], [1, 2])
    assert np.array_equal(dataset.user_history["U2"], [2, padding_row])
    assert np.array_equal(dataset.user_history["U3"], [padding_row, padding_row])


def test_parses_one_line(small_news, write_file):
    dataset = DKNDataset(small_news, write_file("history", ["U3"]), history_size=2)

    assert dataset.parser_one_line("1 U3 N7%imp9") == (1.0, "U3", "N7", "imp9")
    assert dataset.parser_one_line("0 U3 N7") == (0.0, "U3", "N7", 0)


@pytest.mark.parametrize(
    "batch_size, expected_sizes", [(10, [10, 10, 10, 10]), (15, [15, 15, 10])]
)
def test_batches_the_file(synthetic_dkn, batch_size, expected_sizes):
    dataset = DKNDataset(synthetic_dkn["news"], synthetic_dkn["history"], HISTORY_SIZE)
    batches = list(dataset.load_data_from_file(synthetic_dkn["train"], batch_size))

    assert [len(ids) for _, ids in batches] == expected_sizes
    for (np_batch, _), size in zip(batches, expected_sizes):
        assert np_batch["labels"].shape == (size, 1)
        assert np_batch["candidate_words"].shape == (size, DOC_SIZE)
        assert np_batch["candidate_entities"].shape == (size, DOC_SIZE)
        assert np_batch["clicked_words"].shape == (size, HISTORY_SIZE, DOC_SIZE)
        assert np_batch["clicked_entities"].shape == (size, HISTORY_SIZE, DOC_SIZE)


def test_looks_up_the_candidate_and_the_clicked_articles(small_news, write_file):
    history = write_file("history", ["U1 N1,N3", "U2"])
    data = write_file("data", ["1 U1 N2%5", "0 U2 N3%5"])
    dataset = DKNDataset(small_news, history, history_size=2)

    np_batch, impression_ids = next(dataset.load_data_from_file(data, 8))

    assert impression_ids == ["5", "5"]
    assert np.array_equal(np_batch["labels"], [[1.0], [0.0]])
    assert np.array_equal(np_batch["candidate_words"], [[21, 22, 23], [31, 32, 33]])
    assert np.array_equal(np_batch["candidate_entities"], [[0, 3, 0], [4, 4, 0]])
    assert np.array_equal(
        np_batch["clicked_words"],
        [[[11, 12, 13], [31, 32, 33]], [[0, 0, 0], [0, 0, 0]]],
    )
    assert np.array_equal(
        np_batch["clicked_entities"], [[[1, 0, 2], [4, 4, 0]], [[0, 0, 0], [0, 0, 0]]]
    )


def test_emits_the_dtypes_the_model_consumes(synthetic_dkn):
    dataset = DKNDataset(synthetic_dkn["news"], synthetic_dkn["history"], HISTORY_SIZE)
    np_batch, _ = next(dataset.load_data_from_file(synthetic_dkn["train"], 4))

    assert np_batch["labels"].dtype == np.float32
    for key in (
        "candidate_words",
        "candidate_entities",
        "clicked_words",
        "clicked_entities",
    ):
        assert np_batch[key].dtype == np.int64


def test_rejects_a_click_on_an_unknown_article(small_news, write_file):
    history = write_file("history", ["U1 N1,N9"])

    with pytest.raises(KeyError, match="N9"):
        DKNDataset(small_news, history, history_size=2)


# --------------------------- item-to-item ---------------------------


@pytest.mark.parametrize(
    "batch_size, expected_groups", [(5, [5, 5]), (4, [4, 4, 2]), (10, [10])]
)
def test_batches_the_file_in_groups(synthetic_dkn, batch_size, expected_groups):
    dataset = DKNItem2ItemDataset(synthetic_dkn["news"], NEG_NUM)
    batches = list(
        dataset.load_data_from_file(synthetic_dkn["item2item_train"], batch_size)
    )

    group_size = NEG_NUM + 2
    assert [len(ids) for _, ids in batches] == [
        groups * group_size for groups in expected_groups
    ]
    for (np_batch, _), groups in zip(batches, expected_groups):
        assert np_batch["words"].shape == (groups, group_size, DOC_SIZE)
        assert np_batch["entities"].shape == (groups, group_size, DOC_SIZE)
        assert np_batch["words"].dtype == np.int64
        assert np_batch["entities"].dtype == np.int64


def test_keeps_the_file_order_within_a_group(small_news, write_file):
    data = write_file("data", ["N2", "N1", "N3", "N3", "N2", "N1"])
    dataset = DKNItem2ItemDataset(small_news, neg_num=1)

    np_batch, news_ids = next(dataset.load_data_from_file(data, 8))

    assert news_ids == ["N2", "N1", "N3", "N3", "N2", "N1"]
    assert np.array_equal(np_batch["words"][:, :, 0], [[21, 11, 31], [31, 21, 11]])


def test_rejects_an_incomplete_trailing_group(small_news, write_file):
    data = write_file("data", ["N2", "N1", "N3", "N3"])
    dataset = DKNItem2ItemDataset(small_news, neg_num=1)

    with pytest.raises(ValueError, match="incomplete group"):
        list(dataset.load_data_from_file(data, 8))
