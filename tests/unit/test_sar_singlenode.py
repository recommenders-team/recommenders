# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import itertools
import numpy as np
import pandas as pd
from reco_utils.common.constants import PREDICTION_COL
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode
from reco_utils.recommender.sar import TIME_NOW
from tests.sar_common import read_matrix, load_userpred, load_affinity


def _rearrange_to_test(array, row_ids, col_ids, row_map, col_map):
    """Rearranges SAR array into test array order"""
    if row_ids is not None:
        row_index = [row_map[x] for x in row_ids]
        array = array[row_index, :]
    if col_ids is not None:
        col_index = [col_map[x] for x in col_ids]
        array = array[:, col_index]
    return array


def test_init(header):
    model = SARSingleNode(
        remove_seen=True, similarity_type="jaccard", **header
    )

    assert model.col_user == "UserId"
    assert model.col_item == "MovieId"
    assert model.col_rating == "Rating"
    # TODO: add more parameters


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_fit(similarity_type, timedecay_formula, train_test_dummy_timestamp, header):
    model = SARSingleNode(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=timedecay_formula,
        **header
    )
    trainset, testset = train_test_dummy_timestamp
    model.fit(trainset)


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_predict(
    similarity_type, timedecay_formula, train_test_dummy_timestamp, header
):
    model = SARSingleNode(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=timedecay_formula,
        **header
    )
    trainset, testset = train_test_dummy_timestamp
    model.fit(trainset)
    preds = model.predict(testset)

    assert len(preds) == 2
    assert isinstance(preds, pd.DataFrame)
    assert preds[header["col_user"]].dtype == object
    assert preds[header["col_item"]].dtype == object
    assert preds[PREDICTION_COL].dtype == trainset[header["col_rating"]].dtype


@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [
        (1, "cooccurrence", "count"),
        (1, "jaccard", "jac"),
        (1, "lift", "lift"),
        (3, "cooccurrence", "count"),
        (3, "jaccard", "jac"),
        (3, "lift", "lift"),
    ],
)
def test_sar_item_similarity(
    threshold, similarity_type, file, demo_usage_data, sar_settings, header
):

    model = SARSingleNode(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=False,
        time_decay_coefficient=30,
        time_now=TIME_NOW,
        threshold=threshold,
        **header
    )

    model.fit(demo_usage_data)

    true_item_similarity, row_ids, col_ids = read_matrix(
        sar_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    if similarity_type is "cooccurrence":
        test_item_similarity = _rearrange_to_test(
            model.item_similarity.todense(),
            row_ids,
            col_ids,
            model.item2index,
            model.item2index,
        )
        assert np.array_equal(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
        )
    else:
        test_item_similarity = _rearrange_to_test(
            model.item_similarity,
            row_ids,
            col_ids,
            model.item2index,
            model.item2index,
        )
        assert np.allclose(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
            atol=sar_settings["ATOL"],
        )


def test_user_affinity(demo_usage_data, sar_settings, header):
    time_now = demo_usage_data[header["col_timestamp"]].max()
    model = SARSingleNode(
        remove_seen=True,
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        **header
    )
    model.fit(demo_usage_data)

    true_user_affinity, items = load_affinity(sar_settings["FILE_DIR"] + "user_aff.csv")
    user_index = model.user2index[sar_settings["TEST_USER_ID"]]
    test_user_affinity = np.reshape(
        np.array(
            _rearrange_to_test(
                model.user_affinity, None, items, None, model.item2index
            )[user_index,].todense()
        ),
        -1,
    )
    assert np.allclose(
        true_user_affinity.astype(test_user_affinity.dtype),
        test_user_affinity,
        atol=sar_settings["ATOL"],
    )


@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")],
)
def test_userpred(
    threshold, similarity_type, file, header, sar_settings, demo_usage_data
):
    time_now = demo_usage_data[header["col_timestamp"]].max()
    model = SARSingleNode(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        threshold=threshold,
        **header
    )
    model.fit(demo_usage_data)

    true_items, true_scores = load_userpred(
        sar_settings["FILE_DIR"]
        + "userpred_"
        + file
        + str(threshold)
        + "_userid_only.csv"
    )
    test_results = model.recommend_k_items(
        demo_usage_data[
            demo_usage_data[header["col_user"]] == sar_settings["TEST_USER_ID"]
        ],
        top_k=10,
        sort_top_k=True
    )
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=sar_settings["ATOL"])
