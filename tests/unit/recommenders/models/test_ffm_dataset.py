# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

from recommenders.models.deeprec.io.ffm_dataset import FFMDataset

# matches the files written by the synthetic_ffm fixture in conftest.py
FIELD_COUNT = 6


def test_parses_one_line():
    dataset = FFMDataset(FIELD_COUNT)
    label, features, impression_id = dataset.parser_one_line("1 1:5:0.5 3:9:2%imp7")

    assert label == 1.0
    # field and feature indices are 1-based in the file, 0-based in memory
    assert features == [[0, 4, 0.5], [2, 8, 2.0]]
    assert impression_id == "imp7"


def test_parses_line_without_impression_id():
    dataset = FFMDataset(FIELD_COUNT)
    _, _, impression_id = dataset.parser_one_line("0 1:5:1")

    assert impression_id == 0


@pytest.mark.parametrize(
    "batch_size, expected_sizes", [(20, [20, 20]), (30, [30, 10]), (40, [40])]
)
def test_batches_the_file(synthetic_ffm, batch_size, expected_sizes):
    dataset = FFMDataset(FIELD_COUNT)
    batches = list(dataset.load_data_from_file(synthetic_ffm["train"], batch_size))

    assert [len(ids) for _, ids in batches] == expected_sizes
    for (np_batch, _), size in zip(batches, expected_sizes):
        assert np_batch["labels"].shape == (size, 1)
        assert np_batch["dnn_offsets"].shape == (size * FIELD_COUNT,)
        assert np_batch["feat_ids"].shape == np_batch["feat_values"].shape


def test_groups_features_into_one_bag_per_field(synthetic_ffm):
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


def test_sorts_features_by_field(tmp_path):
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


def test_emits_the_dtypes_the_model_consumes(synthetic_ffm):
    dataset = FFMDataset(FIELD_COUNT)
    np_batch, _ = next(
        dataset.load_data_from_file(synthetic_ffm["train"], batch_size=4)
    )

    assert np_batch["labels"].dtype == np.float32
    assert np_batch["feat_ids"].dtype == np.int64
    assert np_batch["feat_values"].dtype == np.float32
    assert np_batch["dnn_offsets"].dtype == np.int64


def test_rejects_field_index_beyond_field_count(tmp_path):
    path = os.path.join(tmp_path, "bad_field")
    with open(path, "w") as f:
        f.write("1 1:11:1 9:91:1\n")

    dataset = FFMDataset(field_count=4)
    with pytest.raises(ValueError, match="field index"):
        next(dataset.load_data_from_file(path, batch_size=1))
