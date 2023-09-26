# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import papermill as pm
import scrapbook as sb


@pytest.mark.notebooks
@pytest.mark.skip(reason="Wikidata API is unstable")
def test_wikidata_runs(notebooks, output_notebook, kernel_name, tmp):
    notebook_path = notebooks["wikidata_knowledge_graph"]
    MOVIELENS_SAMPLE_SIZE = 5
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MOVIELENS_DATA_SIZE="100k",
            MOVIELENS_SAMPLE=True,
            MOVIELENS_SAMPLE_SIZE=MOVIELENS_SAMPLE_SIZE,
        ),
    )


@pytest.mark.notebooks
@pytest.mark.skip(reason="Wikidata API is unstable")
def test_wikidata_values(notebooks, output_notebook, kernel_name):
    notebook_path = notebooks["wikidata_knowledge_graph"]
    pm.execute_notebook(
        notebook_path,
        output_notebook,
        kernel_name=kernel_name,
        parameters=dict(
            MOVIELENS_DATA_SIZE="100k", MOVIELENS_SAMPLE=True, MOVIELENS_SAMPLE_SIZE=5
        ),
    )
    results = sb.read_notebook(output_notebook).scraps.dataframe.set_index("name")[
        "data"
    ]

    # NOTE: The return number should be always 5, but sometimes we get less because wikidata is unstable
    assert results["length_result"] >= 1
