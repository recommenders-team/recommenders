# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from collections import defaultdict

try:
    from recommenders.models.sasrec.model import SASREC
    from recommenders.models.sasrec.ssept import SSEPT
    from recommenders.models.sasrec.sampler import WarpSampler
    from recommenders.models.sasrec.util import SASRecDataSet

    from recommenders.datasets.amazon_reviews import (
        download_and_extract,
        _reviews_preprocessing,
    )

except ImportError:
    pass  # skip if in cpu environment


@pytest.fixture()
def model_parameters():
    params = {
        "itemnum": 85930,
        "usernum": 63114,
        "maxlen": 50,
        "num_blocks": 2,
        "hidden_units": 100,
        "num_heads": 1,
        "dropout_rate": 0.1,
        "l2_emb": 0.0,
        "num_neg_test": 100,
    }
    return params


def data_process_with_time(fname, pname, K=10, sep=" ", item_set=None, add_time=False):
    User = defaultdict(list)
    Users = set()
    Items = set()
    user_dict, item_dict = {}, {}

    item_counter = defaultdict(lambda: 0)
    user_counter = defaultdict(lambda: 0)
    with open(fname, "r") as fr:
        for line in fr:
            u, i, t = line.rstrip().split(sep)
            User[u].append((i, t))
            Items.add(i)
            Users.add(u)
            item_counter[i] += 1
            user_counter[u] += 1

    # remove items with less than K interactions
    print(f"Read {len(User)} users and {len(Items)} items")
    remove_items = set()
    count_remove, count_missing = 0, 0
    for item in Items:
        if item_counter[item] < K:
            count_remove += 1
            remove_items.add(item)
        elif item_set and item not in item_set:
            count_missing += 1
            remove_items.add(item)

    if count_remove > 0:
        print(f"{count_remove} items have less than {K} interactions")

    if count_missing > 0:
        print(f"{count_missing} items are not in the meta data")

    Items = Items - remove_items

    # remove users with less than K interactions
    remove_users = set()
    count_remove = 0
    # Users = set(User.keys())
    for user in Users:
        if user_counter[user] < K:
            remove_users.add(user)
            count_remove += 1
    if count_remove > 0:
        print(f"{count_remove} users have less than {K} interactions")
        Users = Users - remove_users

    print(f"Total {len(Users)} users and {len(Items)} items")
    item_count = 1
    for item in Items:
        item_dict[item] = item_count
        item_count += 1

    count_del = 0
    user_count = 1
    with open(pname, "w") as fw:
        for user in Users:
            items = User[user]
            items = [tup for tup in items if tup[0] in Items]
            if len(items) < K:
                count_del += 1
            else:
                user_dict[user] = user_count
                # sort by time
                items = sorted(items, key=lambda x: x[1])

                # replace by the item-code
                timestamps = [x[1] for x in items]
                items = [item_dict[x[0]] for x in items]
                for i, t in zip(items, timestamps):
                    out_txt = [str(user_count), str(i)]
                    if add_time:
                        out_txt.append(str(t))
                    fw.write(sep.join(out_txt) + "\n")
                user_count += 1

    print(f"Total {user_count-1} users, {count_del} removed")
    print(f"Processed model input data in {pname}")
    return user_dict, item_dict


@pytest.mark.gpu
def test_prepare_data():
    data_dir = os.path.join("..", "..", "tests", "resources", "deeprec", "sasrec")
    dataset = "reviews_Electronics_5"
    reviews_name = dataset + ".json"
    outfile = os.path.join(data_dir, dataset + ".txt")

    reviews_file = os.path.join(data_dir, reviews_name)
    download_and_extract(reviews_name, reviews_file)
    reviews_output = _reviews_preprocessing(reviews_file)
    _, _ = data_process_with_time(reviews_output, outfile, K=10, sep="\t")

    # initiate a dataset class
    data = SASRecDataSet(filename=outfile, col_sep="\t")

    # create train, validation and test splits
    data.split()

    assert len(data.user_train) > 0
    assert len(data.user_valid) > 0
    assert len(data.user_test) > 0


@pytest.mark.gpu
def test_sampler():
    batch_size = 8
    maxlen = 50
    data_dir = os.path.join("..", "..", "tests", "resources", "deeprec", "sasrec")
    dataset = "reviews_Electronics_5"
    reviews_name = dataset + ".json"
    outfile = os.path.join(data_dir, dataset + ".txt")

    reviews_file = os.path.join(data_dir, reviews_name)
    download_and_extract(reviews_name, reviews_file)
    reviews_output = _reviews_preprocessing(reviews_file)
    _, _ = data_process_with_time(reviews_output, outfile, K=10, sep="\t")

    # initiate a dataset class
    data = SASRecDataSet(filename=outfile, col_sep="\t")

    # create train, validation and test splits
    data.split()

    sampler = WarpSampler(
        data.user_train,
        data.usernum,
        data.itemnum,
        batch_size=batch_size,
        maxlen=maxlen,
        n_workers=3,
    )
    u, seq, pos, neg = sampler.next_batch()

    assert len(u) == batch_size
    assert len(seq) == batch_size
    assert len(pos) == batch_size
    assert len(neg) == batch_size


@pytest.mark.gpu
def test_sasrec(model_parameters):

    params = model_parameters

    model = SASREC(
        item_num=params["itemnum"],
        seq_max_len=params["maxlen"],
        num_blocks=params["num_blocks"],
        embedding_dim=params["hidden_units"],
        attention_dim=params["hidden_units"],
        attention_num_heads=params["num_heads"],
        dropout_rate=params["dropout_rate"],
        conv_dims=[100, 100],
        l2_reg=params["l2_emb"],
        num_neg_test=params["num_neg_test"],
    )

    assert model.encoder is not None
    assert model.item_embedding_layer is not None


@pytest.mark.gpu
def test_ssept(model_parameters):

    params = model_parameters

    model = SSEPT(
        item_num=params["itemnum"],
        user_num=params["usernum"],
        seq_max_len=params["maxlen"],
        num_blocks=params["num_blocks"],
        user_embedding_dim=params["hidden_units"],
        item_embedding_dim=params["hidden_units"],
        attention_dim=params["hidden_units"],
        attention_num_heads=params["num_heads"],
        dropout_rate=params["dropout_rate"],
        conv_dims=[200, 200],
        l2_reg=params["l2_emb"],
        num_neg_test=params["num_neg_test"],
    )

    assert model.encoder is not None
    assert model.item_embedding_layer is not None
    assert model.user_embedding_layer is not None
