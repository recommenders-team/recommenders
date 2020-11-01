# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.dataset.mind import download_mind, extract_mind


train_or_valid = ["train", "valid"]

zipsizes = {
    "small": {
        "train": 52952752,
        "valid": 30945572
    }
}

filesizes = {
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
def test_download_mind(tmp):
    train_path, valid_path = download_mind(size="small", dest_path=tmp)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 52952752
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 30945572


@pytest.mark.smoke
def test_extract_mind(tmp):
    for dataset_size in filesizes: #demo or small
        train_zip, valid_zip = download_mind(size=dataset_size, dest_path=tmp)
        unzipped_paths = extract_mind(train_zip, valid_zip)
        assert len(unzipped_paths) == 2
        for pos in range(len(unzipped_paths)): # either train_path or valid_path
            for filename in filesizes[dataset_size][train_or_valid[pos]]:
                statinfo = os.stat(os.path.join(unzipped_paths[pos], filename))
                assert statinfo.st_size == filesizes[dataset_size][train_or_valid[pos]][filename] 
    
