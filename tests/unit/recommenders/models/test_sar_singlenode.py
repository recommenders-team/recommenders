# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import codecs
import csv
import itertools
import json
import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import urllib

from recommenders.utils.constants import DEFAULT_PREDICTION_COL
from recommenders.models.sar.sar_singlenode import SARSingleNode


def _csv_reader_url(url, delimiter=",", encoding="utf-8"):
    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, encoding), delimiter=delimiter)
    return csvfile


def load_affinity(file):
    """Loads user affinities from test dataset"""
    reader = _csv_reader_url(file)
    items = next(reader)[1:]
    affinities = np.array(next(reader)[1:])
    return affinities, items


def load_userpred(file, k=10):
    """Loads test predicted items and their SAR scores"""
    reader = _csv_reader_url(file)
    next(reader)
    values = next(reader)
    items = values[1 : (k + 1)]
    scores = np.array([float(x) for x in values[(k + 1) :]])
    return items, scores


def read_matrix(file, row_map=None, col_map=None):
    """read in test matrix and hash it"""
    reader = _csv_reader_url(file)

    # skip the header
    col_ids = next(reader)[1:]
    row_ids = []
    rows = []
    for row in reader:
        rows += [row[1:]]
        row_ids += [row[0]]
    array = np.array(rows)

    # now map the rows and columns to the right values
    if row_map is not None and col_map is not None:
        row_index = [row_map[x] for x in row_ids]
        col_index = [col_map[x] for x in col_ids]
        array = array[row_index, :]
        array = array[:, col_index]
    return array, row_ids, col_ids


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
    model = SARSingleNode(similarity_type="jaccard", **header)

    assert model.col_user == "UserId"
    assert model.col_item == "MovieId"
    assert model.col_rating == "Rating"
    assert model.col_timestamp == "Timestamp"
    assert model.col_prediction == "prediction"
    assert model.similarity_type == "jaccard"
    assert model.time_decay_half_life == 2592000
    assert not model.time_decay_flag
    assert model.time_now is None
    assert model.threshold == 1


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_fit(similarity_type, timedecay_formula, train_test_dummy_timestamp, header):
    model = SARSingleNode(
        similarity_type=similarity_type, timedecay_formula=timedecay_formula, **header
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
        similarity_type=similarity_type, timedecay_formula=timedecay_formula, **header
    )
    trainset, testset = train_test_dummy_timestamp
    model.fit(trainset)
    preds = model.predict(testset)

    assert len(preds) == 2
    assert isinstance(preds, pd.DataFrame)
    assert preds[header["col_user"]].dtype == trainset[header["col_user"]].dtype
    assert preds[header["col_item"]].dtype == trainset[header["col_item"]].dtype
    assert preds[DEFAULT_PREDICTION_COL].dtype == trainset[header["col_rating"]].dtype


def test_predict_all_items(train_test_dummy_timestamp, header):
    model = SARSingleNode(**header)
    trainset, _ = train_test_dummy_timestamp
    model.fit(trainset)

    user_items = itertools.product(
        trainset[header["col_user"]].unique(), trainset[header["col_item"]].unique()
    )
    testset = pd.DataFrame(user_items, columns=[header["col_user"], header["col_item"]])
    preds = model.predict(testset)

    assert len(preds) == len(testset)
    assert isinstance(preds, pd.DataFrame)
    assert preds[header["col_user"]].dtype == trainset[header["col_user"]].dtype
    assert preds[header["col_item"]].dtype == trainset[header["col_item"]].dtype
    assert preds[DEFAULT_PREDICTION_COL].dtype == trainset[header["col_rating"]].dtype


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
        similarity_type=similarity_type,
        timedecay_formula=False,
        time_decay_coefficient=30,
        threshold=threshold,
        **header
    )

    # Remove duplicates
    demo_usage_data = demo_usage_data.sort_values(
        header["col_timestamp"], ascending=False
    )
    demo_usage_data = demo_usage_data.drop_duplicates(
        [header["col_user"], header["col_item"]],
        keep="first"
    )

    model.fit(demo_usage_data)

    true_item_similarity, row_ids, col_ids = read_matrix(
        sar_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    if similarity_type == "cooccurrence":
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
            model.item_similarity, row_ids, col_ids, model.item2index, model.item2index
        )
        assert np.allclose(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
            atol=sar_settings["ATOL"],
        )


def test_user_affinity(demo_usage_data, sar_settings, header):
    time_now = demo_usage_data[header["col_timestamp"]].max()
    model = SARSingleNode(
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        **header
    )
    model.fit(demo_usage_data)

    true_user_affinity, items = load_affinity(sar_settings["FILE_DIR"] + "user_aff.csv")
    user_index = model.user2index[sar_settings["TEST_USER_ID"]]
    sar_user_affinity = np.reshape(
        np.array(
            _rearrange_to_test(
                model.user_affinity, None, items, None, model.item2index
            )[
                user_index,
            ].todense()
        ),
        -1,
    )
    assert np.allclose(
        true_user_affinity.astype(sar_user_affinity.dtype),
        sar_user_affinity,
        atol=sar_settings["ATOL"],
    )


@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")],
)
def test_recommend_k_items(
    threshold, similarity_type, file, header, sar_settings, demo_usage_data
):
    time_now = demo_usage_data[header["col_timestamp"]].max()
    model = SARSingleNode(
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
        sort_top_k=True,
        remove_seen=True,
    )
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=sar_settings["ATOL"])


def test_get_item_based_topk(header, pandas_dummy):

    sar = SARSingleNode(**header)
    sar.fit(pandas_dummy)

    # test with just items provided
    expected = pd.DataFrame(
        dict(UserId=[0, 0, 0], MovieId=[8, 7, 6], prediction=[2.0, 2.0, 2.0])
    )
    items = pd.DataFrame({header["col_item"]: [1, 5, 10]})
    actual = sar.get_item_based_topk(items, top_k=3)
    assert_frame_equal(expected, actual, check_dtype=False)

    # test with items and users
    expected = pd.DataFrame(
        dict(
            UserId=[100, 100, 100, 1, 1, 1],
            MovieId=[8, 7, 6, 4, 3, 10],
            prediction=[2.0, 2.0, 2.0, 2.0, 2.0, 1.0],
        )
    )
    items = pd.DataFrame(
        {
            header["col_user"]: [100, 100, 1, 100, 1, 1],
            header["col_item"]: [1, 5, 1, 10, 2, 6],
        }
    )
    actual = sar.get_item_based_topk(items, top_k=3, sort_top_k=True)
    assert_frame_equal(expected, actual, check_dtype=False)

    # test with items, users, and ratings
    expected = pd.DataFrame(
        dict(
            UserId=[100, 100, 100, 1, 1, 1],
            MovieId=[2, 4, 3, 4, 3, 10],
            prediction=[5.0, 5.0, 5.0, 8.0, 8.0, 4.0],
        )
    ).set_index(["UserId", "MovieId"])
    items = pd.DataFrame(
        {
            header["col_user"]: [100, 100, 1, 100, 1, 1],
            header["col_item"]: [1, 5, 1, 10, 2, 6],
            header["col_rating"]: [5, 1, 3, 1, 5, 4],
        }
    )
    actual = sar.get_item_based_topk(items, top_k=3).set_index(["UserId", "MovieId"])
    assert_frame_equal(expected, actual, check_like=True)


def test_get_popularity_based_topk(header):

    train_df = pd.DataFrame(
        {
            header["col_user"]: [1, 1, 1, 2, 2, 2, 3, 3, 3],
            header["col_item"]: [1, 2, 3, 1, 3, 4, 5, 6, 1],
            header["col_rating"]: [1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )

    sar = SARSingleNode(**header)
    sar.fit(train_df)

    expected = pd.DataFrame(dict(MovieId=[1, 3, 4], prediction=[3, 2, 1]))
    actual = sar.get_popularity_based_topk(top_k=3, sort_top_k=True)
    assert_frame_equal(expected, actual)


def test_get_normalized_scores(header):
    train = pd.DataFrame(
        {
            header["col_user"]: [1, 1, 1, 1, 2, 2, 2, 2],
            header["col_item"]: [1, 2, 3, 4, 1, 5, 6, 7],
            header["col_rating"]: [3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0],
            header["col_timestamp"]: [1, 20, 30, 400, 50, 60, 70, 800],
        }
    )
    test = pd.DataFrame(
        {
            header["col_user"]: [1, 1, 1, 2, 2, 2],
            header["col_item"]: [5, 6, 7, 2, 3, 4],
            header["col_rating"]: [2.0, 1.0, 5.0, 3.0, 4.0, 5.0],
        }
    )

    model = SARSingleNode(**header, timedecay_formula=True, normalize=True)
    model.fit(train)
    actual = model.score(test, remove_seen=True)
    expected = np.array(
        [
            [-np.inf, -np.inf, -np.inf, -np.inf, 1.23512374, 1.23512374, 1.23512374],
            [-np.inf, 1.23512374, 1.23512374, 1.23512374, -np.inf, -np.inf, -np.inf],
        ]
    )
    assert actual.shape == (2, 7)
    assert isinstance(actual, np.ndarray)
    assert np.isclose(expected, np.asarray(actual)).all()

    actual = model.score(test)
    expected = np.array(
        [
            [
                3.11754872,
                4.29408577,
                4.29408577,
                4.29408577,
                1.23512374,
                1.23512374,
                1.23512374,
            ],
            [
                2.5293308,
                1.23511758,
                1.23511758,
                1.23511758,
                3.11767458,
                3.11767458,
                3.11767458,
            ],
        ]
    )

    assert actual.shape == (2, 7)
    assert isinstance(actual, np.ndarray)
    assert np.isclose(expected, np.asarray(actual)).all()


def test_match_similarity_type(header):
    # store parameters in json
    with open('similarity_type_test.json', 'w') as f:
        json.dump({'similarity_type': 'jaccard'}, f)
    # load parameters in json
    with open('similarity_type_test.json') as f:
        params = json.load(f)

    params.update(header)

    model = SARSingleNode(**params)

    train = pd.DataFrame(
        {
            header["col_user"]: [1, 1, 1, 1, 2, 2, 2, 2],
            header["col_item"]: [1, 2, 3, 4, 1, 5, 6, 7],
            header["col_rating"]: [3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0],
            header["col_timestamp"]: [1, 20, 30, 400, 50, 60, 70, 800],
        }
    )

    model.fit(train)
