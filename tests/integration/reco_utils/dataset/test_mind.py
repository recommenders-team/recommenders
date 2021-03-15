# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import papermill as pm
import scrapbook as sb

from reco_utils.dataset.mind import download_mind, extract_mind
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME


@pytest.mark.integration
def test_download_mind(tmp_path):
    train_path, valid_path = download_mind(size="large", dest_path=tmp_path)
    statinfo = os.stat(train_path)
    assert statinfo.st_size == 530196631
    statinfo = os.stat(valid_path)
    assert statinfo.st_size == 103456245


@pytest.mark.integration
def test_extract_mind(tmp):
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


@pytest.mark.integration
def test_mind_utils_integration(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["mind_utils"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(mind_type="small", word_embedding_dim=300),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    assert results["utils_state"]["vert_num"] == 17
    assert results["utils_state"]["subvert_num"] == 17
    assert results["utils_state"]["word_num"] == 23404
    assert results["utils_state"]["word_num_all"] == 41074
    assert results["utils_state"]["embedding_exist_num"] == 22408
    assert results["utils_state"]["embedding_exist_num_all"] == 37634
    assert results["utils_state"]["uid2index"] == 5000
