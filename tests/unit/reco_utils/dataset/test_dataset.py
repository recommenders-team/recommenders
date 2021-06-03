# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pytest
from tempfile import TemporaryDirectory
import logging
from reco_utils.dataset.download_utils import maybe_download, download_path


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
    assert downloaded_filepath.split("/")[-1] == "license.txt"


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
    assert "File ./license.txt already downloaded" in caplog.text


# def test_maybe_download_retry(caplog):
#     TODO: consider https://github.com/rholder/retrying/blob/master/retrying.py
#     caplog.clear()
#     caplog.set_level(logging.INFO)

#     maybe_download(
#         "https://raw.githubusercontent.com/Microsoft/Recommenders/main/non_existing_file.zip"
#     )
#     assert "Backing off" in caplog.text


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
