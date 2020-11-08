# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import requests
from reco_utils.dataset.mind import download_mind, extract_mind

@pytest.mark.parametrize("url, content_length, etag", 
    [("https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_train.zip",
    "17372879", "0x8D82C63E386D09C"),
    ("https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_dev.zip",
    "10080022", "0x8D82C6434EC3CEE"),
    ("https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_utils.zip", 
    "97292694", "0x8D87F362FF7FB26"),
    ("https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "52952752","0x8D834F2EB31BDEC"),
    ("https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
    "30945572","0x8D834F2EBA8D865"),
    ("https://mind201910small.blob.core.windows.net/release/MINDsmall_utils.zip", 
    "155178106", "0x8D87F67F4AEB960"),
    ("https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
    "530196631","0x8D8244E90C15C07"),
    ("https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
    "103456245","0x8D8244E92005849"),
    ("https://mind201910small.blob.core.windows.net/release/MINDlarge_utils.zip", 
    "150359301", "0x8D87F67E6CA4364"),
    ])
def test_mind_url(url, content_length, etag):
    """ Test file sizes and etags. 
    Covers train, dev and utils files for demo, small and large datasets. 
    """
    url_headers = requests.head(url).headers
    assert url_headers["Content-Length"] == content_length
    assert url_headers["ETag"] == etag

@pytest.mark.parametrize("size",[("demo"),("small")])
def test_extract_mind(size,tmp):
    """ Test file download and extration for demo and small datasets """
    train_zip, valid_zip = download_mind(size, dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip)

    if size == "demo":
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
    elif size == "small":
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
    else:
        assert False
        