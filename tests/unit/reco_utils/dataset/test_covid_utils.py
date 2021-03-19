# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from unittest.mock import patch, MagicMock
import pytest
from reco_utils.dataset.covid_utils import (
    remove_duplicates,
    remove_nan,
    clean_dataframe,
    retrieve_text,
    get_public_domain_text,
)

import numpy as np
import pandas as pd


@pytest.fixture(scope="module")
def df():
    mock_metadata = {
        "cord_uid": ["ej795nks", "", np.nan, "adygntbe", "adygntbe"],
        "doi": [
            "10.1289/ehp.7117",
            np.nan,
            "10.1371/journal.pmed.0030149",
            "",
            "10.1016/s0140-6736(03)13507-6",
        ],
        "title": [
            "Understanding the Spatial Clustering of",
            "The Application of the Haddon Matrix to",
            "Cynomolgus Macaque as an Animal Model for",
            "SARS: screening, disease associations",
            "SARS: screening, disease associations",
        ],
        "license": ["cc0", "cc0", "cc0", "no-cc", "els-covid"],
        "url": [
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13",
            "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7",
            "https://doi.org/10.1016/s0140-6736(03)13507-6",
        ],
    }
    return pd.DataFrame(mock_metadata)


def test_remove_duplicates(df):
    output = remove_duplicates(df, cols=["cord_uid", "doi", "title", "license", "url"])
    assert True not in output.duplicated(["cord_uid"]).values


def test_remove_nan(df):
    output = remove_nan(df, cols=["cord_uid", "doi", "title", "license", "url"])
    assert np.nan not in output["cord_uid"].values


def test_clean_dataframe(df):
    output = clean_dataframe(df)
    assert len(df) > len(output)


def test_retrieve_text():
    def mock_get(uri, headers):
        class MockResponse:
            def json(self):
                return dict(body_text=[dict(text="test")])

        return MockResponse()

    with patch("reco_utils.dataset.covid_utils.requests.get", side_effect=mock_get):
        result = retrieve_text(entry=dict(pdf_json_files="a"), container_name="test")
    assert "test" == result


def test_get_public_domain_text(df):
    df["publish_time"] = ""
    df["authors"] = ""
    df["journal"] = ""
    df["abstract"] = ""

    def mock_retrieve_text(
        row, container_name, azure_storage_account_name, azure_storage_sas_token
    ):
        return "full text"

    with patch(
        "reco_utils.dataset.covid_utils.retrieve_text", side_effect=mock_retrieve_text
    ):
        full = get_public_domain_text(df, container_name="test")

    assert all(full["full_text"] == ["full text"] * 5)
