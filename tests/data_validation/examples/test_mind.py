# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from recommenders.utils.notebook_utils import execute_notebook, read_notebook


def test_mind_utils_runs(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["mind_utils"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(mind_type="small", word_embedding_dim=300),
    )


def test_mind_utils_values(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["mind_utils"]
    execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(mind_type="demo", word_embedding_dim=300),
    )
    results = read_notebook(output_notebook)

    assert results["vert_num"] == 17
    assert results["subvert_num"] == 17
    assert results["word_num"] == 23404
    assert results["word_num_all"] == 41074
    assert results["embedding_exist_num"] == 22408
    assert results["embedding_exist_num_all"] == 37634
    assert results["uid2index"] == 5000
