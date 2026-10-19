# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pickle

import numpy as np
import pytest

from recommenders.models.newsrec.io.mind_dataset import MINDDataset


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@pytest.mark.gpu
def test_mind_dataset_titles(mind_demo):
    dataset = MINDDataset(*mind_demo["train"], load(mind_demo["word_dict"]), 30, 50)
    batch = next(dataset.load_train_batches(32, 4, np.random.default_rng(0)))
    assert dataset.news["title"].shape[1] == 30
    assert batch["candidate_rows"].shape == (32, 5)
    assert batch["clicked_rows"].shape == (32, 50)
    assert batch["users"].shape == (32,)


@pytest.mark.gpu
def test_mind_dataset_all_views(mind_demo):
    dataset = MINDDataset(
        *mind_demo["valid"],
        load(mind_demo["word_dict_all"]),
        30,
        50,
        uid2index=load(mind_demo["uid2index"]),
        body_size=50,
        vert_dict=load(mind_demo["vert_dict"]),
        subvert_dict=load(mind_demo["subvert_dict"]),
    )
    news_count = len(dataset.news["title"])
    assert dataset.news["body"].shape == (news_count, 50)
    assert dataset.news["vert"].shape == dataset.news["subvert"].shape == (news_count,)
    assert len(dataset.impressions) == len(dataset.labels) == len(dataset.users)
