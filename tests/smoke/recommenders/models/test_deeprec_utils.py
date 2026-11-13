# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pickle

import numpy as np
import pytest

try:
    import tensorflow as tf
    from recommenders.models.deeprec.deeprec_utils import (
        prepare_hparams,
        download_deeprec_resources,
    )
    from recommenders.models.deeprec.io.dkn_dataset import (
        DKNDataset,
        DKNItem2ItemDataset,
    )
    from recommenders.models.deeprec.io.sequential_iterator import SequentialIterator
except ImportError:
    pass  # disable error while collecting tests for non-gpu environments


@pytest.mark.gpu
def test_dkn_dataset(deeprec_resource_path):
    data_path = os.path.join(deeprec_resource_path, "dkn")
    data_file = os.path.join(data_path, "train_mind_demo.txt")
    news_feature_file = os.path.join(data_path, "doc_feature.txt")
    download_deeprec_resources(
        "https://raw.githubusercontent.com/recommenders-team/resources/main/deeprec/",
        data_path,
        "mind-demo.zip",
    )

    dataset = DKNDataset(
        news_feature_file, os.path.join(data_path, "user_history.txt"), 50
    )
    n_instances = 0
    for batch, impression_ids in dataset.load_data_from_file(data_file, 100):
        size = len(impression_ids)
        assert batch["labels"].shape == (size, 1)
        assert batch["candidate_rows"].shape == (size,)
        assert batch["clicked_rows"].shape == (size, 50)
        n_instances += size
    with open(data_file) as rd:
        assert n_instances == sum(1 for _ in rd)

    # doc_list.txt holds 20 news IDs: 4 groups of neg_num + 2 = 5.
    dataset_item2item = DKNItem2ItemDataset(news_feature_file, neg_num=3)
    batches = list(
        dataset_item2item.load_data_from_file(
            os.path.join(data_path, "doc_list.txt"), 3
        )
    )
    assert [batch["words"].shape for batch, _ in batches] == [(3, 5, 10), (1, 5, 10)]
    assert [len(news_ids) for _, news_ids in batches] == [15, 5]


@pytest.fixture
def sequential_data(tmp_path):
    paths = {}
    for name, vocab in [
        ("user_vocab", {"U1": 1, "U2": 2}),
        ("item_vocab", {"I1": 1, "I2": 2, "I3": 3, "I4": 4}),
        ("cate_vocab", {"C1": 1, "C2": 2}),
    ]:
        path = os.path.join(tmp_path, name + ".pkl")
        with open(path, "wb") as f:
            pickle.dump(vocab, f)
        paths[name] = path

    train_file = os.path.join(tmp_path, "train_data")
    with open(train_file, "w") as f:
        f.write("1\tU1\tI2\tC2\t259200\tI1\tC1\t86400\n")
        f.write("0\tU2\tI4\tC2\t864000\tI1,I2,I3\tC1,C2,C1\t86400,259200,518400\n")
        f.write(
            "1\tunknown\tunknown\tunknown\t1296000\tI1,I2,I3,I4"
            "\tC1,C2,C1,C2\t86400,259200,518400,864000\n"
        )
    return train_file, paths


@pytest.mark.gpu
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_Sequential_Iterator(sequential_data, deeprec_config_path, batch_size):
    train_file, paths = sequential_data
    yaml_file = os.path.join(deeprec_config_path, "sli_rec.yaml")
    hparams = prepare_hparams(
        yaml_file, **paths, batch_size=batch_size, max_seq_length=3
    )
    iterator = SequentialIterator(hparams, tf.Graph())
    assert iterator is not None
    batches = list(iterator.load_data_from_file(train_file))
    assert len(batches) == (3 + batch_size - 1) // batch_size
    for i, res in enumerate(batches):
        assert isinstance(res, dict)
        assert res[iterator.labels].shape == (min(batch_size, 3 - i * batch_size), 1)

    for tensor, expected in [
        (iterator.labels, [[1], [0], [1]]),
        (iterator.users, [1, 2, 0]),
        (iterator.items, [2, 4, 0]),
        (iterator.cates, [2, 2, 0]),
        (iterator.item_history, [[1, 0, 0], [1, 2, 3], [2, 3, 4]]),
        (iterator.item_cate_history, [[1, 0, 0], [1, 2, 1], [2, 1, 2]]),
        (iterator.mask, [[1, 0, 0], [1, 1, 1], [1, 1, 1]]),
        (iterator.time, [259200, 864000, 1296000]),
    ]:
        np.testing.assert_array_equal(
            np.concatenate([res[tensor] for res in batches]), expected
        )

    for tensor, expected in [
        (iterator.time_diff, [[2, 1, 1], [2, 3, 4], [3, 4, 5]]),
        (iterator.time_from_first_action, [[2, 1, 1], [2, 5, 9], [5, 9, 14]]),
        (iterator.time_to_now, [[2, 1, 1], [9, 7, 4], [12, 9, 5]]),
    ]:
        np.testing.assert_allclose(
            np.concatenate([res[tensor] for res in batches]), np.log(expected)
        )
