# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

FFM_FIELD_COUNT = 6
FFM_SPLITS = [("train", 40), ("valid", 20), ("test", 20)]


@pytest.fixture(scope="module")
def synthetic_ffm(tmp_path_factory):
    """Tiny synthetic FFM files, so the tests need no download.

    Every line carries exactly one feature per field, with the 1-based field and
    feature indices the format uses, and alternating 0/1 labels.
    """
    d = tmp_path_factory.mktemp("ffm")
    rng = np.random.RandomState(0)
    paths = {}
    for name, n_lines in FFM_SPLITS:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            for i in range(n_lines):
                parts = [str(i % 2)]
                for field in range(1, FFM_FIELD_COUNT + 1):
                    feature = field * 20 + rng.randint(1, 11)
                    parts.append("{0}:{1}:1".format(field, feature))
                f.write(" ".join(parts) + "\n")
        paths[name] = path
    return paths


DKN_NEWS_COUNT = 30
DKN_DOC_SIZE = 5
DKN_WORD_COUNT = 40
DKN_ENTITY_COUNT = 15
DKN_DIM = 8
DKN_ENTITY_DIM = 6
DKN_USER_COUNT = 8
DKN_SPLITS = [("train", 40), ("valid", 20), ("test", 20)]
DKN_IMPRESSION_SIZE = 4
DKN_ITEM2ITEM_GROUPS = [("item2item_train", 10), ("item2item_valid", 5)]
DKN_ITEM2ITEM_GROUP_SIZE = 4


@pytest.fixture(scope="module")
def synthetic_dkn(tmp_path_factory):
    """Tiny synthetic DKN files, so the tests need no download.

    30 articles of 5 words and 5 entities; 8 users where user ``U<u>`` clicked
    ``u`` articles, so histories run from empty to longer than the tests'
    history size; word, entity and context embeddings; data files with
    alternating 0/1 labels, the valid and test lines grouped into impressions of
    4; and item-to-item files in groups of 4 (``neg_num = 2``).
    """
    d = tmp_path_factory.mktemp("dkn")
    rng = np.random.RandomState(0)
    paths = {}
    news_ids = ["N{0}".format(i) for i in range(1, DKN_NEWS_COUNT + 1)]

    paths["news"] = os.path.join(d, "doc_feature.txt")
    with open(paths["news"], "w") as f:
        for news_id in news_ids:
            words = rng.randint(1, DKN_WORD_COUNT, DKN_DOC_SIZE)
            entities = rng.randint(0, DKN_ENTITY_COUNT, DKN_DOC_SIZE)
            f.write(
                "{0} {1} {2}\n".format(
                    news_id, ",".join(map(str, words)), ",".join(map(str, entities))
                )
            )

    paths["history"] = os.path.join(d, "user_history.txt")
    with open(paths["history"], "w") as f:
        for user in range(DKN_USER_COUNT):
            line = "U{0}".format(user)
            if user:
                line += " " + ",".join(rng.choice(news_ids, user, replace=False))
            f.write(line + "\n")

    for name, shape in [
        ("word", (DKN_WORD_COUNT, DKN_DIM)),
        ("entity", (DKN_ENTITY_COUNT, DKN_ENTITY_DIM)),
        ("context", (DKN_ENTITY_COUNT, DKN_ENTITY_DIM)),
    ]:
        paths[name] = os.path.join(d, name + ".npy")
        np.save(paths[name], rng.normal(scale=0.1, size=shape).astype(np.float32))

    for name, n_lines in DKN_SPLITS:
        paths[name] = os.path.join(d, name)
        with open(paths[name], "w") as f:
            for i in range(n_lines):
                line = "{0} U{1} {2}".format(
                    i % 2, rng.randint(DKN_USER_COUNT), rng.choice(news_ids)
                )
                if name != "train":
                    line += "%{0}".format(i // DKN_IMPRESSION_SIZE)
                f.write(line + "\n")

    for name, n_groups in DKN_ITEM2ITEM_GROUPS:
        paths[name] = os.path.join(d, name)
        with open(paths[name], "w") as f:
            for _ in range(n_groups):
                group = rng.choice(news_ids, DKN_ITEM2ITEM_GROUP_SIZE, replace=False)
                f.write("\n".join(group) + "\n")

    return paths
