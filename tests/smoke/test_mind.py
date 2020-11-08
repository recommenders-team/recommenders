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

