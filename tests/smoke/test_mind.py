# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import os
from reco_utils.dataset.mind import download_mind, extract_mind


@pytest.mark.smoke
def test_download_mind(tmp):
    train_path, valid_path = download_mind(size="small", dest_path=tmp)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 52952752
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 30945572


@pytest.mark.smoke
def test_extract_mind(tmp):
    train_zip, valid_zip = download_mind(size="small", dest_path=tmp)
    train_path, valid_path = extract_mind(train_zip, valid_zip)

    statinfo = os.stat(os.path.join(train_path, "behaviors.tsv"))
    assert statinfo.st_size == 92047111
    statinfo = os.stat(os.path.join(train_path, "entity_embedding.vec"))
    assert statinfo.st_size == 25811015
    statinfo = os.stat(os.path.join(train_path, "news.tsv"))
    assert statinfo.st_size == 45895926
    statinfo = os.stat(os.path.join(train_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588

    statinfo = os.stat(os.path.join(valid_path, "behaviors.tsv"))
    assert statinfo.st_size == 42975799
    statinfo = os.stat(os.path.join(valid_path, "entity_embedding.vec"))
    assert statinfo.st_size == 21960998
    statinfo = os.stat(os.path.join(valid_path, "news.tsv"))
    assert statinfo.st_size == 37410117
    statinfo = os.stat(os.path.join(valid_path, "relation_embedding.vec"))
    assert statinfo.st_size == 1044588
