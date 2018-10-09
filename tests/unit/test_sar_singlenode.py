import sys
import os
import itertools
import pytest
from sklearn.model_selection import train_test_split
import pandas as pd

# TODO: better solution??
root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)
)
sys.path.append(root)
from utilities.recommender.sar.sar_singlenode import SARSingleNodeReference


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

