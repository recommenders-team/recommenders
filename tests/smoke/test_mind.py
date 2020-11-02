# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
import requests
from reco_utils.dataset.mind import download_mind, extract_mind



train_or_valid = ["train", "valid"]

# zipsizes = {
#     "small": {
#         "train": 52952752,
#         "valid": 30945572
#     }
# }





# @pytest.mark.smoke
# def test_download_mind(tmp):
#     train_path, valid_path = download_mind(size="small", dest_path=tmp)
#     statinfo = os.stat(train_path)
#     assert statinfo.st_size == 52952752
#     statinfo = os.stat(valid_path)
#     assert statinfo.st_size == 30945572


URL_INFO = {
    "large": {
        "train": {
            "url": "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip",
            "headers": {
                'Content-Length': '530196631',
                'ETag': '0x8D8244E90C15C07'
            }
        },
        "valid": {
            "url": "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip",
            "headers": {
                'Content-Length': '103456245',
                'ETag': '0x8D8244E92005849'
            }
        },
    },
    "small": {
        "train": {
            "url": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
            "headers": {
                'Content-Length': '52952752',
                'ETag': '0x8D834F2EB31BDEC'
            }
        },
        "valid": {
            "url": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
            "headers": {
                'Content-Length': '30945572',
                'ETag': '0x8D834F2EBA8D865'
            }
        }
    },
    "demo": {
        "train": {
            "url": "https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_train.zip",
            "headers": {
                'Content-Length': '17372879',
                'ETag': '0x8D82C63E386D09C'
            }
        },
        "valid": {
            "url": "https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_dev.zip",
            "headers": {
                'Content-Length': '10080022',
                'ETag': '0x8D82C6434EC3CEE'
            }
        },
        "utils": {
            "url": "https://recodatasets.blob.core.windows.net/newsrec/MINDdemo_utils.zip",
            "headers": {
                'Content-Length': '97292490',
                'ETag': '0x8D82C66AF165260'
            }
        }
    }
}

@pytest.mark.smoke
def test_mind_urls():
    """ Test headers (without downloading) of all 7 URLs for demo, small and large for file size and ETag """
    for mind_size in URL_INFO:
        for dataset_type in URL_INFO[mind_size]:
            url_headers = requests.head(URL_INFO[mind_size][dataset_type]["url"]).headers
            for hdr in URL_INFO[mind_size][dataset_type]["headers"]:
                assert url_headers[hdr] == URL_INFO[mind_size][dataset_type]["headers"][hdr]



FILESIZES = {
    "demo":{
        "train":{
            "behaviors.tsv": 14707247,
            "entity_embedding.vec": 16077470,
            "news.tsv": 23120370,
            "relation_embedding.vec": 1044588
        },
        "valid":{
            "behaviors.tsv": 4434762,
            "entity_embedding.vec": 11591565,
            "news.tsv": 15624320,
            "relation_embedding.vec": 1044588
        }
    },
    "small":{
        "train":{
            "behaviors.tsv": 92019716,
            "entity_embedding.vec": 25811015,
            "news.tsv": 41202121,
            "relation_embedding.vec": 1044588
        },
        "valid":{
            "behaviors.tsv": 42838544,
            "entity_embedding.vec": 21960998,
            "news.tsv": 33519092,
            "relation_embedding.vec": 1044588
        }
    }
}

@pytest.mark.smoke
def test_extract_mind(tmp):
    """ Only extract demo and small train and valid files """
    for dataset_size in FILESIZES: #demo or small
        train_zip, valid_zip = download_mind(size=dataset_size, dest_path=tmp)
        unzipped_paths = extract_mind(train_zip, valid_zip)
        assert len(unzipped_paths) == 2
        for pos in range(len(unzipped_paths)): # either train_path or valid_path
            for filename in FILESIZES[dataset_size][train_or_valid[pos]]:
                statinfo = os.stat(os.path.join(unzipped_paths[pos], filename))
                assert statinfo.st_size == FILESIZES[dataset_size][train_or_valid[pos]][filename] 
    
