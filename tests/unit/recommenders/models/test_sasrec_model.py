# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import os
import pytest
from collections import defaultdict

try:
    import torch
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
    data_dir = os.path.join("tests", "resources", "deeprec", "sasrec")
    dataset = "reviews_Electronics_5"
    reviews_name = dataset + ".json"
    outfile = os.path.join(data_dir, dataset + ".txt")

    reviews_file = os.path.join(data_dir, reviews_name)
    download_and_extract(reviews_name, reviews_file)
    reviews_output = _reviews_preprocessing(reviews_file)
    _, _ = data_process_with_time(reviews_output, outfile, K=10, sep="\t")

    # initiate a dataset class
    data = SASRecDataSet(filename=outfile, col_sep="\t")

    # create train, validation and test splits with explicit parameters
    stats = data.split(valid_size=1, test_size=1, min_interactions=3)

    assert len(data.user_train) > 0
    assert len(data.user_valid) > 0
    assert len(data.user_test) > 0

    # verify stats are returned
    assert "num_users" in stats
    assert "num_items" in stats
    assert "users_with_splits" in stats
    assert "users_train_only" in stats
    assert stats["num_users"] == data.usernum
    assert stats["num_items"] == data.itemnum


@pytest.fixture
def temp_data_file(tmp_path):
    """Create a temporary data file for testing split functionality."""
    # Create test data with known structure:
    # User 1: 5 items (should have train/valid/test)
    # User 2: 3 items (should have train/valid/test with min_interactions=3)
    # User 3: 2 items (should only have train with min_interactions=3)
    data = [
        "1\t101",
        "1\t102",
        "1\t103",
        "1\t104",
        "1\t105",
        "2\t201",
        "2\t202",
        "2\t203",
        "3\t301",
        "3\t302",
    ]
    filepath = tmp_path / "test_data.txt"
    filepath.write_text("\n".join(data))
    return str(filepath)


def test_split_default_parameters(temp_data_file):
    """Test split with default parameters."""
    data = SASRecDataSet(filename=temp_data_file, col_sep="\t")
    stats = data.split()

    # Default: valid_size=1, test_size=1, min_interactions=3
    assert stats["valid_size"] == 1
    assert stats["test_size"] == 1
    assert stats["min_interactions"] == 3

    # User 1 (5 items): train=3, valid=1, test=1
    assert len(data.user_train[1]) == 3
    assert len(data.user_valid[1]) == 1
    assert len(data.user_test[1]) == 1
    assert data.user_valid[1] == [104]
    assert data.user_test[1] == [105]

    # User 2 (3 items): train=1, valid=1, test=1
    assert len(data.user_train[2]) == 1
    assert len(data.user_valid[2]) == 1
    assert len(data.user_test[2]) == 1

    # User 3 (2 items): train=2, valid=0, test=0 (below min_interactions)
    assert len(data.user_train[3]) == 2
    assert len(data.user_valid[3]) == 0
    assert len(data.user_test[3]) == 0


def test_split_custom_sizes(temp_data_file):
    """Test split with custom valid_size and test_size."""
    data = SASRecDataSet(filename=temp_data_file, col_sep="\t")
    stats = data.split(valid_size=2, test_size=1, min_interactions=4)

    # Verify custom parameters are reflected in stats
    assert stats["valid_size"] == 2
    assert stats["test_size"] == 1
    assert stats["min_interactions"] == 4

    # User 1 (5 items): train=2, valid=2, test=1
    assert len(data.user_train[1]) == 2
    assert len(data.user_valid[1]) == 2
    assert len(data.user_test[1]) == 1
    assert data.user_train[1] == [101, 102]
    assert data.user_valid[1] == [103, 104]
    assert data.user_test[1] == [105]

    # User 2 (3 items): below min_interactions=4, train only
    assert len(data.user_train[2]) == 3
    assert len(data.user_valid[2]) == 0
    assert len(data.user_test[2]) == 0


def test_split_returns_stats(temp_data_file):
    """Test that split returns correct statistics."""
    data = SASRecDataSet(filename=temp_data_file, col_sep="\t")
    stats = data.split(valid_size=1, test_size=1, min_interactions=3)

    assert stats["num_users"] == 3
    assert stats["num_items"] == 302  # max item ID (items are 101-105, 201-203, 301-302)
    assert stats["users_with_splits"] == 2  # Users 1 and 2
    assert stats["users_train_only"] == 1  # User 3


def test_split_invalid_min_interactions(temp_data_file):
    """Test that split raises error for invalid min_interactions."""
    data = SASRecDataSet(filename=temp_data_file, col_sep="\t")

    # min_interactions must be >= valid_size + test_size + 1
    with pytest.raises(ValueError, match="min_interactions"):
        data.split(valid_size=2, test_size=2, min_interactions=3)  # 3 < 2+2+1=5


def test_split_no_filename():
    """Test that split raises error when no filename is set."""
    data = SASRecDataSet()

    with pytest.raises(ValueError, match="Filename is required"):
        data.split()


def test_split_verbose(temp_data_file, capsys):
    """Test that verbose=True prints statistics."""
    data = SASRecDataSet(filename=temp_data_file, col_sep="\t")
    data.split(verbose=True)

    captured = capsys.readouterr()
    assert "Split complete" in captured.out
    assert "Total users" in captured.out
    assert "Total items" in captured.out


@pytest.mark.gpu
def test_sampler():
    batch_size = 8
    maxlen = 50
    data_dir = os.path.join("tests", "resources", "deeprec", "sasrec")
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

    # Test that the model is a PyTorch module
    assert isinstance(model, torch.nn.Module)

    # Test forward pass with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 2
    seq_len = params["maxlen"]
    dummy_input = {
        "input_seq": torch.randint(0, 100, (batch_size, seq_len)).to(device),
        "positive": torch.randint(0, 100, (batch_size, seq_len)).to(device),
        "negative": torch.randint(0, 100, (batch_size, seq_len)).to(device),
    }

    model.train()
    pos_logits, neg_logits, istarget = model(dummy_input, training=True)

    assert pos_logits.shape == (batch_size * seq_len, 1)
    assert neg_logits.shape == (batch_size * seq_len, 1)
    assert istarget.shape == (batch_size * seq_len,)


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

    # Test that the model is a PyTorch module
    assert isinstance(model, torch.nn.Module)

    # Test forward pass with dummy data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch_size = 2
    seq_len = params["maxlen"]
    dummy_input = {
        "users": torch.randint(1, 100, (batch_size, 1)).to(device),
        "input_seq": torch.randint(0, 100, (batch_size, seq_len)).to(device),
        "positive": torch.randint(0, 100, (batch_size, seq_len)).to(device),
        "negative": torch.randint(0, 100, (batch_size, seq_len)).to(device),
    }

    model.train()
    pos_logits, neg_logits, istarget = model(dummy_input, training=True)

    assert pos_logits.shape == (batch_size * seq_len, 1)
    assert neg_logits.shape == (batch_size * seq_len, 1)
    assert istarget.shape == (batch_size * seq_len,)
