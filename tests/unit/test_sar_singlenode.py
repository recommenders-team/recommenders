import sys
import os
import itertools
import pytest
from sklearn.model_selection import train_test_split
import time
import datetime
import numpy as np
import pandas as pd
import urllib.request
import csv
import codecs
from utilities.common.constants import PREDICTION_COL
from utilities.recommender.sar.sar_singlenode import SARSingleNodeReference
from utilities.recommender.sar import TIME_NOW


# absolute tolerance parameter for matrix equivalnce in SAR tests
ATOL = 1e-8
# directory of the current file - used to link unit test data
FILE_DIR = "http://recodatasets.blob.core.windows.net/sarunittest/"
# user ID used in the test files (they are designed for this user ID, this is part of the test)
TEST_USER_ID = "0003000098E85347"


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


@pytest.fixture
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


@pytest.fixture
def load_affinity(file):
    """Loads user affinities from test dataset"""
    reader = _csv_reader_url(file)
    items = next(reader)[1:]
    affinities = np.array(next(reader)[1:])
    return affinities, items


@pytest.fixture
def load_userped(file, k=10):
    """Loads test predicted items and their SAR scores"""
    reader = _csv_reader_url(file)
    next(reader)
    values = next(reader)
    items = values[1 : (k + 1)]
    scores = np.array([float(x) for x in values[(k + 1) :]])
    return items, scores


@pytest.fixture
def load_demo_usage_data(header):
    # load the data
    data = pd.read_csv(FILE_DIR + "demoUsage.csv")
    data["rating"] = pd.Series([1] * data.shape[0])
    data = data.rename(
        columns={
            "userId": header["col_user"],
            "productId": header["col_item"],
            "rating": header["col_rating"],
            "timestamp": header["col_timestamp"],
        }
    )

    # convert timestamp
    data[header["col_timestamp"]] = data[header["col_timestamp"]].apply(
        lambda s: time.mktime(
            datetime.datetime.strptime(s, "%Y/%m/%dT%H:%M:%S").timetuple()
        )
    )

    return data


def _sar_hash(train, test, header, pandas_new=False):
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
    # Actually, this is an ordered set - it's discrete, but contiguous.
    # This helps keep the matrices we keep in memory as small as possible.
    enumerate_items_1, enumerate_items_2 = itertools.tee(enumerate(unique_items))
    enumerate_users_1, enumerate_users_2 = itertools.tee(enumerate(unique_users))
    item_map_dict = {x: i for i, x in enumerate_items_1}
    user_map_dict = {x: i for i, x in enumerate_users_1}

    # the reverse of the dictionary above - array index to actual ID
    index2user = dict(enumerate_users_2)
    index2item = dict(enumerate_items_2)

    return (
        unique_users,
        unique_items,
        user_map_dict,
        item_map_dict,
        index2user,
        index2item,
    )


def test_init(header):
    model = SARSingleNodeReference(
        remove_seen=True, similarity_type="jaccard", **header
    )

    assert model.col_user == "UserId"
    assert model.col_item == "MovieId"
    assert model.col_rating == "Rating"
    # TODO: add more parameters


@pytest.mark.parametrize("similarity_type, timedecay_formula", [
    ("jaccard", False),
    ("lift", True),
])
def test_fit(similarity_type, timedecay_formula, get_train_test, header):
    model = SARSingleNodeReference(
        remove_seen=True, similarity_type=similarity_type, timedecay_formula=timedecay_formula, **header
    )
    trainset, testset = get_train_test
    unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = _sar_hash(
        trainset, testset, header
    )
    model.set_index(
        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item
    )
    model.fit(trainset)


@pytest.mark.parametrize("similarity_type, timedecay_formula", [
    ("jaccard", False),
    ("lift", True),
])
def test_predict(similarity_type, timedecay_formula, get_train_test, header):
    model = SARSingleNodeReference(
        remove_seen=True, similarity_type=similarity_type, timedecay_formula=timedecay_formula, **header
    )
    trainset, testset = get_train_test

    unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = _sar_hash(
        trainset, testset, header
    )
    model.set_index(
        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item
    )
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
    threshold, similarity_type, file, load_demo_usage_data, header
):
    data = load_demo_usage_data
    
    model = SARSingleNodeReference(
            remove_seen=True,
            similarity_type=similarity_type,
            timedecay_formula=False,
            time_decay_coefficient=30,
            time_now=TIME_NOW,
            threshold=threshold,
            **header
        )

    unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = _sar_hash(
        data, None, header
    )

    # we need to index the train and test sets for SAR matrix operations to work
    model.set_index(
        unique_users,
        unique_items,
        user_map_dict,
        item_map_dict,
        index2user,
        index2item,
    )
    model.fit(data)
   
    true_item_similarity, row_ids, col_ids = read_matrix(
        FILE_DIR + "sim_" + file + str(threshold) + ".csv"
    )

    if similarity_type is "cooccurrence":
        test_item_similarity = rearrange_to_test(
            model.item_similarity.todense(),
            row_ids,
            col_ids,
            item_map_dict,
            item_map_dict,
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
            item_map_dict,
            item_map_dict,
        )
        assert np.allclose(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
            atol=ATOL,
        )


# Test 7
# def test_user_affinity():
#     data = load_demo_usage_data()
#     time_now = data[header()["col_timestamp"]].max()
#     tester = setup_SAR(
#         data,
#         similarity_type="cooccurrence",
#         timedecay_formula=True,
#         time_now=time_now,
#         time_decay_coefficient=30,
#     )
#     true_user_affinity, items = load_affinity(FILE_DIR + "user_aff.csv")
#     user_index = tester.user_map_dict[TEST_USER_ID]
#     test_user_affinity = np.reshape(
#         np.array(
#             rearrange_to_test(
#                 tester.model.user_affinity, None, items, None, tester.item_map_dict
#             )[user_index,].todense()
#         ),
#         -1,
#     )
#     assert np.allclose(
#         true_user_affinity.astype(test_user_affinity.dtype),
#         test_user_affinity,
#         atol=ATOL,
#     )


# Tests 8-10
params = "threshold,similarity_type,file"


# @pytest.mark.parametrize(
#     params, [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")]
# )
# def test_userpred(threshold, similarity_type, file):
#     data = load_demo_usage_data()
#     time_now = data[header()["col_timestamp"]].max()
#     tester = setup_SAR(
#         data,
#         remove_seen=True,
#         similarity_type=similarity_type,
#         timedecay_formula=True,
#         time_now=time_now,
#         time_decay_coefficient=30,
#         threshold=threshold,
#     )
#     true_items, true_scores = load_userped(
#         FILE_DIR + "userpred_" + file + str(threshold) + "_userid_only.csv"
#     )
#     test_results = tester.model.recommend_k_items(
#         data[data[header()["col_user"]] == TEST_USER_ID], top_k=10
#     )
#     test_items = list(test_results[header()["col_item"]])
#     test_scores = np.array(test_results["prediction"])
#     assert true_items == test_items
#     assert np.allclose(true_scores, test_scores, atol=ATOL)
