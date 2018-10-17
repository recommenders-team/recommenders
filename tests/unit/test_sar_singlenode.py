import sys
import os
import itertools
import pytest
from sklearn.model_selection import train_test_split
import time
import datetime
import numpy as np
import pandas as pd
import csv
import urllib.request
import codecs

# TODO: better solution??
from utilities.common.constants import PREDICTION_COL

root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
sys.path.append(root)
from utilities.recommender.sar.sar_singlenode import SARSingleNodeReference
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from utilities.recommender.sar import TIME_NOW

def csv_reader_url(file, delimiter=",", encoding="utf-8"):
    """
    Read a csv file over http

    Returns:
         csv reader iterable
    """
    ftpstream = urllib.request.urlopen(file)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, encoding), delimiter=delimiter)
    return csvfile

# convenient way to invoke SAR with different parameters on different datasets
# TODO: pytest class fixtures are not yet supported as of this release
# @pytest.fixture
class setup_SAR:
    def __init__(
        self,
        data,
        header,
        remove_seen=True,
        similarity_type="jaccard",
        timedecay_formula=False,
        time_decay_coefficient=30,
        threshold=1,
        time_now=TIME_NOW
    ):

        self.data = data
        model = SARSingleNodeReference(
            remove_seen=remove_seen,
            similarity_type=similarity_type,
            timedecay_formula=timedecay_formula,
            time_decay_coefficient=time_decay_coefficient,
            time_now=time_now,
            threshold=threshold,
            **header
        )

        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = airship_hash(
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

        self.model = model
        self.user_map_dict = user_map_dict
        self.item_map_dict = item_map_dict


@pytest.fixture(scope="module")
def get_train_test(load_pandas_dummy_timestamp_dataset):
    trainset, testset = train_test_split(
        load_pandas_dummy_timestamp_dataset, test_size=0.2, random_state=0
    )
    return trainset, testset


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


def test_fit(get_train_test, header):
    model = SARSingleNodeReference(
        remove_seen=True, similarity_type="jaccard", **header
    )
    train, test = get_train_test
    unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = _sar_hash(
        train, test, header
    )
    model.set_index(
        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item
    )
    model.fit(train)


"""
Fixtures to load and reconcile custom output from TLC
"""


def read_matrix(file, row_map=None, col_map=None):
    """read in test matrix and hash it"""
    reader = csv_reader_url(file)
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
    reader = csv_reader_url(file)
    items = next(reader)[1:]
    affinities = np.array(next(reader)[1:])
    return affinities, items


@pytest.fixture
def load_userped(file, k=10):
    """Loads test predicted items and their SAR scores"""
    reader = csv_reader_url(file)
    next(reader)
    values = next(reader)
    items = values[1 : (k + 1)]
    scores = np.array([float(x) for x in values[(k + 1) :]])
    return items, scores


@pytest.fixture
def demo_data(header, spark_test_settings):
    # load the data
    data = pd.read_csv(spark_test_settings["FILE_DIR"] + "demoUsage.csv")
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


"""
Other fixtures are below
"""


@pytest.fixture
def load_dataset(load_pandas_dummy_timestamp_dataset):
    """Load a fixture dataset"""
    trainset, testset = sklearn_train_test_split(
        load_pandas_dummy_timestamp_dataset,
        test_size=None,
        train_size=0.8,
        random_state=0,
    )

    return trainset, testset


@pytest.fixture
def sar_algo(header):
    """Add different SAR algos"""
    return [
        [
            SARSingleNodeReference(
                remove_seen=True, similarity_type="jaccard", **header
            ),
            "sar_ref",
        ],
        [
            SARSingleNodeReference(
                remove_seen=True,
                similarity_type="jaccard",
                time_decay_coefficient=30,
                timedecay_formula=True,
                **header
            ),
            "sar_ref",
        ],
    ]


def airship_hash(train, test, header, pandas_new=False):
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


def test_predict(header, sar_algo, load_dataset):
    """Test algo predict"""
    trainset, testset = load_dataset

    unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item = airship_hash(
        trainset, testset, header
    )

    for recommender in sar_algo:
        recommender[0].set_index(
            unique_users,
            unique_items,
            user_map_dict,
            item_map_dict,
            index2user,
            index2item,
        )
        recommender[0].fit(trainset)
        preds = recommender[0].predict(testset)
        assert len(preds) == 2
        assert isinstance(preds, pd.DataFrame)
        assert preds[header["col_user"]].dtype == object
        assert preds[header["col_item"]].dtype == object
        assert preds[PREDICTION_COL].dtype == float


"""
Main SAR tests are below - load test files which are used for both Scala SAR and Python reference implementations
"""

# Tests 1-6
params = "threshold,similarity_type,file"


@pytest.mark.parametrize(
    params,
    [
        (1, "cooccurrence", "count"),
        (1, "jaccard", "jac"),
        (1, "lift", "lift"),
        (3, "cooccurrence", "count"),
        (3, "jaccard", "jac"),
        (3, "lift", "lift"),
    ],
)
def test_sar_item_similarity(threshold, similarity_type, file, demo_data, header, spark_test_settings):
    tester = setup_SAR(demo_data, header, similarity_type=similarity_type, threshold=threshold)
    true_item_similarity, row_ids, col_ids = read_matrix(
        spark_test_settings["FILE_DIR"] + "sim_" + file + str(threshold) + ".csv"
    )

    if similarity_type is "cooccurrence":
        test_item_similarity = rearrange_to_test(
            tester.model.item_similarity.todense(),
            row_ids,
            col_ids,
            tester.item_map_dict,
            tester.item_map_dict,
        )
        assert np.array_equal(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
        )
    else:
        test_item_similarity = rearrange_to_test(
            np.array(tester.model.item_similarity),
            row_ids,
            col_ids,
            tester.item_map_dict,
            tester.item_map_dict,
        )
        assert np.allclose(
            true_item_similarity.astype(test_item_similarity.dtype),
            test_item_similarity,
            atol=spark_test_settings["ATOL"],
        )


# Test 7
def test_user_affinity(demo_data, header, spark_test_settings):
    time_now = demo_data[header["col_timestamp"]].max()
    tester = setup_SAR(
        demo_data,
        header,
        similarity_type="cooccurrence",
        timedecay_formula=True,
        time_now=time_now,
        time_decay_coefficient=30,
    )
    true_user_affinity, items = load_affinity(spark_test_settings["FILE_DIR"] + "user_aff.csv")
    user_index = tester.user_map_dict[spark_test_settings["TEST_USER_ID"]]
    test_user_affinity = np.reshape(
        np.array(
            rearrange_to_test(
                tester.model.user_affinity, None, items, None, tester.item_map_dict
            )[user_index,].todense()
        ),
        -1,
    )
    assert np.allclose(
        true_user_affinity.astype(test_user_affinity.dtype),
        test_user_affinity,
        atol=spark_test_settings["ATOL"],
    )


# Tests 8-10
params = "threshold,similarity_type,file"


@pytest.mark.parametrize(
    params, [(3, "cooccurrence", "count"), (3, "jaccard", "jac"), (3, "lift", "lift")]
)
def test_userpred(threshold, similarity_type, file, demo_data, header, spark_test_settings):
    time_now = demo_data[header["col_timestamp"]].max()
    tester = setup_SAR(
        demo_data,
        header,
        remove_seen=True,
        similarity_type=similarity_type,
        timedecay_formula=True,
        time_now=time_now,
        time_decay_coefficient=30,
        threshold=threshold,
    )
    true_items, true_scores = load_userped(
        spark_test_settings["FILE_DIR"] + "userpred_" + file + str(threshold) + "_userid_only.csv"
    )
    test_results = tester.model.recommend_k_items(
        demo_data[demo_data[header["col_user"]] == spark_test_settings["TEST_USER_ID"]], top_k=10
    )
    test_items = list(test_results[header["col_item"]])
    test_scores = np.array(test_results["prediction"])
    assert true_items == test_items
    assert np.allclose(true_scores, test_scores, atol=spark_test_settings["ATOL"])
