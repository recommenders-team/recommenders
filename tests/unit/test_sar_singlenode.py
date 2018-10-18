import sys
import os
import itertools
import pytest
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import urllib.request
import csv
import codecs
from utilities.common.constants import PREDICTION_COL
from utilities.recommender.sar.sar_singlenode import SARSingleNodeReference
from utilities.recommender.sar import TIME_NOW


def _csv_reader_url(url, delimiter=",", encoding="utf-8"):
    """
    Read a csv file over http

    Returns:
         csv reader iterable
    """
    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, encoding), delimiter=delimiter)
    return csvfile


@pytest.fixture(scope="module")
def get_train_test(load_pandas_dummy_timestamp_dataset):
    trainset, testset = train_test_split(
        load_pandas_dummy_timestamp_dataset, test_size=0.2, random_state=0
    )
    return trainset, testset


def _read_matrix(file, row_map=None, col_map=None):
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


@pytest.fixture
def rearrange_to_test(array, row_ids, col_ids, row_map, col_map):
    """Rearranges SAR array into test array order"""
    if row_ids is not None:
        row_index = [row_map[x] for x in row_ids]
        array = array[row_index, :]
    if col_ids is not None:
        col_index = [col_map[x] for x in col_ids]
        array = array[:, col_index]
    return array


def _load_affinity(file):
    """Loads user affinities from test dataset"""
    reader = _csv_reader_url(file)
    items = next(reader)[1:]
    affinities = np.array(next(reader)[1:])
    return affinities, items


def _load_userped(file, k=10):
    """Loads test predicted items and their SAR scores"""
    reader = _csv_reader_url(file)
    next(reader)
    values = next(reader)
    items = values[1 : (k + 1)]
    scores = np.array([float(x) for x in values[(k + 1) :]])
    return items, scores


def _apply_sar_hash_index(model, train, test, header, pandas_new=False):
    # TODO: review this function
    # index all users and items which SAR will compute scores for
    # bugfix to get around different pandas vesions in build servers
    if test is not None:
        if pandas_new:
            df_all = pd.concat([train, test], sort=False)
        else:
            df_all = pd.concat([train, test])
    else:
        df_all = train

    # hash SAR
    # Obtain all the users and items from both training and test data
    unique_users = df_all[header["col_user"]].unique()
    unique_items = df_all[header["col_item"]].unique()

    # Hash users and items to smaller continuous space.
    # Actually, this is an ordered set - it's discrete, but .
    # This helps keep the matrices we keep in memory as small as possible.
    enumerate_items_1, enumerate_items_2 = itertools.tee(enumerate(unique_items))
    enumerate_users_1, enumerate_users_2 = itertools.tee(enumerate(unique_users))
    item_map_dict = {x: i for i, x in enumerate_items_1}
    user_map_dict = {x: i for i, x in enumerate_users_1}

    # the reverse of the dictionary above - array index to actual ID
    index2user = dict(enumerate_users_2)
    index2item = dict(enumerate_items_2)

    model.set_index(
        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item
    )


def test_init(header):
    model = SARSingleNodeReference(
        remove_seen=True, similarity_type="jaccard", **header
    )

    assert model.col_user == "UserId"
    assert model.col_item == "MovieId"
    assert model.col_rating == "Rating"
    # TODO: add more parameters


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_fit(similarity_type, timedecay_formula, get_train_test, header):
    model = SARSingleNodeReference(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=timedecay_formula,
        **header
    )
    trainset, testset = get_train_test
    _apply_sar_hash_index(model, trainset, testset, header)

    model.fit(trainset)


@pytest.mark.parametrize(
    "similarity_type, timedecay_formula", [("jaccard", False), ("lift", True)]
)
def test_predict(similarity_type, timedecay_formula, get_train_test, header):
    model = SARSingleNodeReference(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=timedecay_formula,
        **header
    )
    trainset, testset = get_train_test

    _apply_sar_hash_index(model, trainset, testset, header)

    model.fit(trainset)
    preds = model.predict(testset)

    assert len(preds) == 2
    assert isinstance(preds, pd.DataFrame)
    assert preds[header["col_user"]].dtype == object
    assert preds[header["col_item"]].dtype == object
    assert preds[PREDICTION_COL].dtype == float


"""
Main SAR tests are below - load test files which are used for both Scala SAR and Python reference implementations
"""

# Tests 1-6
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
    threshold, similarity_type, file, demo_usage_data, sar_test_settings, header
):

    model = SARSingleNodeReference(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=False,
        time_decay_coefficient=30,
        time_now=TIME_NOW,
        threshold=threshold,
        **header
    )

    _apply_sar_hash_index(model, demo_usage_data, None, header)

    model.fit(demo_usage_data)

    true_item_similarity, row_ids, col_ids = _read_matrix(
        sar_test_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    if similarity_type is "cooccurrence":
        test_item_similarity = rearrange_to_test(
            model.item_similarity.todense(),
            row_ids,
            col_ids,
            model.item_map_dict,
            model.item_map_dict,
        )
        assert np.array_equal(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
        )
    else:
        test_item_similarity = rearrange_to_test(
            np.array(model.item_similarity),
            row_ids,
            col_ids,
            model.item_map_dict,
            model.item_map_dict,
        )
        assert np.allclose(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
            atol=sar_test_settings["ATOL"],
        )


# Test 7
def test_user_affinity(demo_usage_data, sar_test_settings, header):
    time_now = demo_usage_data[header["col_timestamp"]].max()
    model = SARSingleNodeReference(
        remove_seen=True,
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        **header
    )
    _apply_sar_hash_index(model, demo_usage_data, None, header)
    model.fit(demo_usage_data)

    true_user_affinity, items = _load_affinity(
        sar_test_settings["FILE_DIR"] + "user_aff.csv"
    )
    user_index = model.user_map_dict[sar_test_settings["TEST_USER_ID"]]
    test_user_affinity = np.reshape(
        np.array(
            rearrange_to_test(
                model.user_affinity, None, items, None, model.item_map_dict
            )[user_index,].todense()
        ),
        -1,
    )
    assert np.allclose(
        true_user_affinity.astype(test_user_affinity.dtype),
        test_user_affinity,
        atol=sar_test_settings["ATOL"],
    )


# Tests 8-10
@pytest.mark.parametrize(
    "threshold,similarity_type,file",
    [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")],
)
def test_userpred(
    threshold, similarity_type, file, header, sar_test_settings, demo_usage_data
):
    time_now = demo_usage_data[header["col_timestamp"]].max()
    model = SARSingleNodeReference(
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=True,
        time_decay_coefficient=30,
        time_now=time_now,
        threshold=threshold,
        **header
    )
    _apply_sar_hash_index(model, demo_usage_data, None, header)
    model.fit(demo_usage_data)

    true_items, true_scores = _load_userped(
        sar_test_settings["FILE_DIR"]
        + "userpred_"
        + file
        + str(threshold)
        + "_userid_only.csv"
    )
    test_results = model.recommend_k_items(
        demo_usage_data[
            demo_usage_data[header["col_user"]] == sar_test_settings["TEST_USER_ID"]
        ],
        top_k=10,
    )
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=sar_test_settings["ATOL"])

