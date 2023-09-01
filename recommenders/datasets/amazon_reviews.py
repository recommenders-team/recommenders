# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import shutil
import pandas as pd
import gzip
import random
import logging
import _pickle as cPickle

from recommenders.utils.constants import SEED
from recommenders.datasets.download_utils import maybe_download


random.seed(SEED)
logger = logging.getLogger()


def get_review_data(reviews_file):
    """Downloads amazon review data (only), prepares in the required format
    and stores in the same location

    Args:
        reviews_file (str): Filename for downloaded reviews dataset.
    """
    reviews_name = reviews_file.split("/")[-1]  # *.json (for url)
    download_and_extract(reviews_name, reviews_file)
    reviews_output = _reviews_preprocessing(reviews_file)
    return reviews_output


def data_preprocessing(
    reviews_file,
    meta_file,
    train_file,
    valid_file,
    test_file,
    user_vocab,
    item_vocab,
    cate_vocab,
    sample_rate=0.01,
    valid_num_ngs=4,
    test_num_ngs=9,
    is_history_expanding=True,
):
    """Create data for training, validation and testing from original dataset

    Args:
        reviews_file (str): Reviews dataset downloaded from former operations.
        meta_file (str): Meta dataset downloaded from former operations.
    """
    reviews_output = _reviews_preprocessing(reviews_file)
    meta_output = _meta_preprocessing(meta_file)
    instance_output = _create_instance(reviews_output, meta_output)
    _create_item2cate(instance_output)
    sampled_instance_file = _get_sampled_data(instance_output, sample_rate=sample_rate)
    preprocessed_output = _data_processing(sampled_instance_file)
    if is_history_expanding:
        _data_generating(preprocessed_output, train_file, valid_file, test_file)
    else:
        _data_generating_no_history_expanding(
            preprocessed_output, train_file, valid_file, test_file
        )
    _create_vocab(train_file, user_vocab, item_vocab, cate_vocab)
    _negative_sampling_offline(
        sampled_instance_file, valid_file, test_file, valid_num_ngs, test_num_ngs
    )


def _create_vocab(train_file, user_vocab, item_vocab, cate_vocab):

    f_train = open(train_file, "r")

    user_dict = {}
    item_dict = {}
    cat_dict = {}

    logger.info("vocab generating...")
    for line in f_train:
        arr = line.strip("\n").split("\t")
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[5]
        cat_list = arr[6]

        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(","):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1
        for c in cat_list.split(","):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

    sorted_user_dict = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_item_dict = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

    uid_voc = {}
    index = 0
    for key, value in sorted_user_dict:
        uid_voc[key] = index
        index += 1

    mid_voc = {}
    mid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_item_dict:
        mid_voc[key] = index
        index += 1

    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    cPickle.dump(uid_voc, open(user_vocab, "wb"))
    cPickle.dump(mid_voc, open(item_vocab, "wb"))
    cPickle.dump(cat_voc, open(cate_vocab, "wb"))


def _negative_sampling_offline(
    instance_input_file, valid_file, test_file, valid_neg_nums=4, test_neg_nums=49
):

    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_input_file, sep="\t", names=columns)
    items_with_popular = list(ns_df["item_id"])

    global item2cate

    # valid negative sampling
    logger.info("start valid negative sampling")
    with open(valid_file, "r") as f:
        valid_lines = f.readlines()
    write_valid = open(valid_file, "w")
    for line in valid_lines:
        write_valid.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < valid_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = neg_item
            words[3] = item2cate[neg_item]
            write_valid.write("\t".join(words) + "\n")

    # test negative sampling
    logger.info("start test negative sampling")
    with open(test_file, "r") as f:
        test_lines = f.readlines()
    write_test = open(test_file, "w")
    for line in test_lines:
        write_test.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < test_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = neg_item
            words[3] = item2cate[neg_item]
            write_test.write("\t".join(words) + "\n")


def _data_generating(input_file, train_file, valid_file, test_file, min_sequence=1):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
        elif tfile == "valid":
            fo = f_valid
        elif tfile == "test":
            fo = f_test
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            history_clk_num = len(movie_id_list)
            cat_str = ""
            mid_str = ""
            dt_str = ""
            for c1 in cate_list:
                cat_str += c1 + ","
            for mid in movie_id_list:
                mid_str += mid + ","
            for dt_time in dt_list:
                dt_str += dt_time + ","
            if len(cat_str) > 0:
                cat_str = cat_str[:-1]
            if len(mid_str) > 0:
                mid_str = mid_str[:-1]
            if len(dt_str) > 0:
                dt_str = dt_str[:-1]
            if history_clk_num >= min_sequence:
                fo.write(
                    line_split[1]
                    + "\t"
                    + user_id
                    + "\t"
                    + movie_id
                    + "\t"
                    + category
                    + "\t"
                    + date_time
                    + "\t"
                    + mid_str
                    + "\t"
                    + cat_str
                    + "\t"
                    + dt_str
                    + "\n"
                )
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)


def _data_generating_no_history_expanding(
    input_file, train_file, valid_file, test_file, min_sequence=1
):
    """Produce train, valid and test file from processed_output file
    Each user's behavior sequence will only produce one line in train file.
    Like, user's behavior sequence: 12345, and this function will write into train file: 12345
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")

    last_user_id = None
    last_movie_id = None
    last_category = None
    last_datetime = None
    last_tfile = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if last_tfile == "train":
            fo = f_train
        elif last_tfile == "valid":
            fo = f_valid
        elif last_tfile == "test":
            fo = f_test
        if user_id != last_user_id or tfile == "valid" or tfile == "test":
            if last_user_id is not None:
                history_clk_num = len(
                    movie_id_list  # noqa: F821 undefined name 'movie_id_list'
                )
                cat_str = ""
                mid_str = ""
                dt_str = ""
                for c1 in cate_list[:-1]:  # noqa: F821 undefined name 'cate_list'
                    cat_str += c1 + ","
                for mid in movie_id_list[  # noqa: F821 undefined name 'movie_id_list'
                    :-1
                ]:
                    mid_str += mid + ","
                for dt_time in dt_list[:-1]:  # noqa: F821 undefined name 'dt_list'
                    dt_str += dt_time + ","
                if len(cat_str) > 0:
                    cat_str = cat_str[:-1]
                if len(mid_str) > 0:
                    mid_str = mid_str[:-1]
                if len(dt_str) > 0:
                    dt_str = dt_str[:-1]
                if history_clk_num > min_sequence:
                    fo.write(
                        line_split[1]
                        + "\t"
                        + last_user_id
                        + "\t"
                        + last_movie_id
                        + "\t"
                        + last_category
                        + "\t"
                        + last_datetime
                        + "\t"
                        + mid_str
                        + "\t"
                        + cat_str
                        + "\t"
                        + dt_str
                        + "\n"
                    )
            if tfile == "train" or last_user_id is None:
                movie_id_list = []
                cate_list = []
                dt_list = []
        last_user_id = user_id
        last_movie_id = movie_id
        last_category = category
        last_datetime = date_time
        last_tfile = tfile
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)


def _create_item2cate(instance_file):
    logger.info("creating item2cate dict")
    global item2cate
    instance_df = pd.read_csv(
        instance_file,
        sep="\t",
        names=["label", "user_id", "item_id", "timestamp", "cate_id"],
    )
    item2cate = instance_df.set_index("item_id")["cate_id"].to_dict()


def _get_sampled_data(instance_file, sample_rate):
    logger.info("getting sampled data...")
    global item2cate
    output_file = instance_file + "_" + str(sample_rate)
    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_file, sep="\t", names=columns)
    items_num = ns_df["item_id"].nunique()
    items_with_popular = list(ns_df["item_id"])
    items_sample, count = set(), 0
    while count < int(items_num * sample_rate):
        random_item = random.choice(items_with_popular)
        if random_item not in items_sample:
            items_sample.add(random_item)
            count += 1
    ns_df_sample = ns_df[ns_df["item_id"].isin(items_sample)]
    ns_df_sample.to_csv(output_file, sep="\t", index=None, header=None)
    return output_file


def _meta_preprocessing(meta_readfile):
    logger.info("start meta preprocessing...")
    meta_writefile = meta_readfile + "_output"
    meta_r = open(meta_readfile, "r")
    meta_w = open(meta_writefile, "w")
    for line in meta_r:
        line_new = eval(line)
        meta_w.write(line_new["asin"] + "\t" + line_new["categories"][0][-1] + "\n")
    meta_r.close()
    meta_w.close()
    return meta_writefile


def _reviews_preprocessing(reviews_readfile):
    logger.info("start reviews preprocessing...")
    reviews_writefile = reviews_readfile + "_output"
    reviews_r = open(reviews_readfile, "r")
    reviews_w = open(reviews_writefile, "w")
    for line in reviews_r:
        line_new = eval(line.strip())
        reviews_w.write(
            str(line_new["reviewerID"])
            + "\t"
            + str(line_new["asin"])
            + "\t"
            + str(line_new["unixReviewTime"])
            + "\n"
        )
    reviews_r.close()
    reviews_w.close()
    return reviews_writefile


def _create_instance(reviews_file, meta_file):
    logger.info("start create instances...")
    dirs, _ = os.path.split(reviews_file)
    output_file = os.path.join(dirs, "instance_output")

    f_reviews = open(reviews_file, "r")
    user_dict = {}
    item_list = []
    for line in f_reviews:
        line = line.strip()
        reviews_things = line.split("\t")
        if reviews_things[0] not in user_dict:
            user_dict[reviews_things[0]] = []
        user_dict[reviews_things[0]].append((line, float(reviews_things[-1])))
        item_list.append(reviews_things[1])

    f_meta = open(meta_file, "r")
    meta_dict = {}
    for line in f_meta:
        line = line.strip()
        meta_things = line.split("\t")
        if meta_things[0] not in meta_dict:
            meta_dict[meta_things[0]] = meta_things[1]

    f_output = open(output_file, "w")
    for user_behavior in user_dict:
        sorted_user_behavior = sorted(user_dict[user_behavior], key=lambda x: x[1])
        for line, _ in sorted_user_behavior:
            user_things = line.split("\t")
            asin = user_things[1]
            if asin in meta_dict:
                f_output.write("1" + "\t" + line + "\t" + meta_dict[asin] + "\n")
            else:
                f_output.write("1" + "\t" + line + "\t" + "default_cat" + "\n")

    f_reviews.close()
    f_meta.close()
    f_output.close()
    return output_file


def _data_processing(input_file):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")
    user_count = {}
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    f_input.seek(0)
    i = 0
    last_user = None
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user == last_user:
            if i < user_count[user] - 2:
                f_output.write("train" + "\t" + line + "\n")
            elif i < user_count[user] - 1:
                f_output.write("valid" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 2:
                f_output.write("train" + "\t" + line + "\n")
            elif i < user_count[user] - 1:
                f_output.write("valid" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        i += 1
    return output_file


def download_and_extract(name, dest_path):
    """Downloads and extracts Amazon reviews and meta datafiles if they don’t already exist

    Args:
        name (str): Category of reviews.
        dest_path (str): File path for the downloaded file.

    Returns:
        str: File path for the extracted file.
    """
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    file_path = os.path.join(dirs, name)
    if not os.path.exists(file_path):
        _download_reviews(name, dest_path)
        _extract_reviews(file_path, dest_path)

    return file_path


def _download_reviews(name, dest_path):
    """Downloads Amazon reviews datafile.

    Args:
        name (str): Category of reviews
        dest_path (str): File path for the downloaded file
    """

    url = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        + name
        + ".gz"
    )

    dirs, file = os.path.split(dest_path)
    maybe_download(url, file + ".gz", work_directory=dirs)


def _extract_reviews(file_path, zip_path):
    """Extract Amazon reviews and meta datafiles from the raw zip files.

    To extract all files,
    use ZipFile's extractall(path) instead.

    Args:
        file_path (str): Destination path for datafile
        zip_path (str): zipfile path
    """
    with gzip.open(zip_path + ".gz", "rb") as zf, open(file_path, "wb") as f:
        shutil.copyfileobj(zf, f)
