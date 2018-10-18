import itertools
import numpy as np
import pandas as pd
import urllib.request
import csv
import codecs


def _index_and_fit(spark, model, df_all, header):

    df_all.createOrReplaceTempView("df_all")

    # create new index for the items
    query = (
        "select "
        + header["col_user"]
        + ", "
        + "dense_rank() over(partition by 1 order by "
        + header["col_user"]
        + ") as row_id, "
        + header["col_item"]
        + ", "
        + "dense_rank() over(partition by 1 order by "
        + header["col_item"]
        + ") as col_id, "
        + header["col_rating"]
        + ", "
        + header["col_timestamp"]
        + " from df_all"
    )
    df_all = spark.sql(query)
    df_all.createOrReplaceTempView("df_all")

    # Obtain all the users and items from both training and test data
    unique_users = np.array(
        [
            x[header["col_user"]]
            for x in df_all.select(header["col_user"]).distinct().toLocalIterator()
        ]
    )
    unique_items = np.array(
        [
            x[header["col_item"]]
            for x in df_all.select(header["col_item"]).distinct().toLocalIterator()
        ]
    )

    # index all rows and columns, then split again intro train and test
    # We perform the reduction on Spark across keys before calling .collect so this is scalable
    index2user = dict(
        df_all.select(["row_id", header["col_user"]])
        .rdd.reduceByKey(lambda _, v: v)
        .collect()
    )
    index2item = dict(
        df_all.select(["col_id", header["col_item"]])
        .rdd.reduceByKey(lambda _, v: v)
        .collect()
    )

    # reverse the dictionaries: actual IDs to inner index
    user_map_dict = {v: k for k, v in index2user.items()}
    item_map_dict = {v: k for k, v in index2item.items()}

    # we need to index the train and test sets for SAR matrix operations to work
    model.set_index(
        unique_users, unique_items, user_map_dict, item_map_dict, index2user, index2item
    )

    model.fit(df_all)

    return df_all


# TODO: DRY with _rearrange_to_test_sql
def _rearrange_to_test(array, row_ids, col_ids, row_map, col_map):
    """Rearranges SAR array into test array order"""
    if row_ids is not None:
        row_index = [row_map[x] for x in row_ids]
        array = array[row_index, :]
    if col_ids is not None:
        col_index = [col_map[x] for x in col_ids]
        array = array[:, col_index]
    return array


def _rearrange_to_test_sql(array, row_ids, col_ids, row_map, col_map):
    """Rearranges SAR array into test array order
    Same as rearrange_to_test but offsets the count by -1 to account for SQL counts starting at 1"""
    if row_ids is not None:
        row_index = [row_map[x] - 1 for x in row_ids]
        array = array[row_index, :]
    if col_ids is not None:
        col_index = [col_map[x] - 1 for x in col_ids]
        array = array[:, col_index]
    return array


def _csv_reader_url(url, delimiter=",", encoding="utf-8"):
    """
    Read a csv file over http

    Returns:
         csv reader iterable
    """
    ftpstream = urllib.request.urlopen(url)
    csvfile = csv.reader(codecs.iterdecode(ftpstream, encoding), delimiter=delimiter)
    return csvfile


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

