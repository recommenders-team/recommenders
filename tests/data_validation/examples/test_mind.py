# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pytest
import papermill as pm
import scrapbook as sb


def test_mind_utils_runs(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["mind_utils"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(mind_type="small", word_embedding_dim=300),
    )


def test_mind_utils_values(notebooks, output_notebook, kernel_name, tmp):
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
