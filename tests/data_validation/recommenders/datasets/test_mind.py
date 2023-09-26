# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pytest
import requests

from recommenders.datasets.mind import download_mind, extract_mind


@pytest.mark.parametrize(
    "url, content_length, etag",
    [
        (
            "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip",
            "17372879",
            '"0x8D8B8AD5B233930"',
        ),  # NOTE: the z20 blob returns the etag with ""
        (
            "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip",
            "10080022",
            '"0x8D8B8AD5B188839"',
        ),
        (
            "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_utils.zip",
            "97292694",
            '"0x8D8B8AD5B126C3B"',
        ),
        (
            "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
            "52952752",
            "0x8D834F2EB31BDEC",
        ),
        (
            "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
            "30945572",
            "0x8D834F2EBA8D865",
        ),
        (
            "https://mind201910small.blob.core.windows.net/release/MINDsmall_utils.zip",
            "155178106",
            "0x8D87F67F4AEB960",
        ),
        (
            "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
            "530196631",
            "0x8D8244E90C15C07",
        ),
        (
            "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
            "103456245",
            "0x8D8244E92005849",
        ),
        (
            "https://mind201910small.blob.core.windows.net/release/MINDlarge_utils.zip",
            "150359301",
            "0x8D87F67E6CA4364",
        ),
    ],
)
def test_mind_url(url, content_length, etag):
    url_headers = requests.head(url).headers
    assert url_headers["Content-Length"] == content_length
    assert url_headers["ETag"] == etag


def test_download_mind_demo(tmp):
    train_path, valid_path = download_mind(size="demo", dest_path=tmp)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 17372879
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 10080022


def test_download_mind_small(tmp):
    train_path, valid_path = download_mind(size="small", dest_path=tmp)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 52952752
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 30945572


def test_extract_mind_demo(tmp):
    train_zip, valid_zip = download_mind(size="demo", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip, clean_zip_file=False)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 14707247
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 16077470
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 23120370
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 4434762
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 11591565
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 15624320
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588


def test_extract_mind_small(tmp):
    train_zip, valid_zip = download_mind(size="small", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip, clean_zip_file=False)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 92019716
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 25811015
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 41202121
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 42838544
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 21960998
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 33519092
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588


def test_download_mind_large(tmp_path):
    train_path, valid_path = download_mind(size="large", dest_path=tmp_path)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 530196631
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 103456245


def test_extract_mind_large(tmp):
    train_zip, valid_zip = download_mind(size="large", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 1373844151
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 40305151
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 84881998
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588

    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 230662527
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 31958202
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 59055351
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
