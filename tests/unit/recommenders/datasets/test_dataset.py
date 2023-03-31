# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
import requests
from tempfile import TemporaryDirectory
import logging
from recommenders.datasets.download_utils import maybe_download, download_path


@pytest.fixture
def files_fixtures():
    file_url = "https://raw.githubusercontent.com/Microsoft/Recommenders/main/LICENSE"
    filepath = "license.txt"
    return file_url, filepath


def test_maybe_download(files_fixtures):
    file_url, filepath = files_fixtures
    if os.path.exists(filepath):
        os.remove(filepath)

    downloaded_filepath = maybe_download(file_url, "license.txt", expected_bytes=1162)
    assert os.path.exists(downloaded_filepath)
    assert os.path.basename(downloaded_filepath) == "license.txt"


def test_maybe_download_wrong_bytes(caplog, files_fixtures):
    caplog.clear()
    caplog.set_level(logging.INFO)

    file_url, filepath = files_fixtures
    if os.path.exists(filepath):
        os.remove(filepath)

    with pytest.raises(IOError):
        filepath = maybe_download(file_url, "license.txt", expected_bytes=0)
        assert "Failed to verify license.txt" in caplog.text


def test_maybe_download_maybe(caplog, files_fixtures):
    caplog.clear()
    caplog.set_level(logging.INFO)

    file_url, filepath = files_fixtures
    if os.path.exists(filepath):
        os.remove(filepath)

    downloaded_filepath = maybe_download(file_url, "license.txt")
    assert os.path.exists(downloaded_filepath)
    maybe_download(file_url, "license.txt")
    assert "File ." + os.path.sep + "license.txt already downloaded" in caplog.text



def test_maybe_download_retry(caplog):
    caplog.clear()
    caplog.set_level(logging.INFO)
    with pytest.raises(requests.exceptions.HTTPError):
        maybe_download(
            "https://recodatasets.z20.web.core.windows.net/non_existing_file.zip"
        )
        assert "Problem downloading" in caplog.text


def test_download_path():
    # Check that the temporal path is created and deleted
    with download_path() as path:
        assert os.path.isdir(path)
    assert not os.path.isdir(path)

    # Check the behavior when a path is provided
    tmp_dir = TemporaryDirectory()
    with download_path(tmp_dir.name) as path:
        assert os.path.isdir(path)
    assert os.path.isdir(path)
