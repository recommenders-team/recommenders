# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import time
import zipfile
import pytest
import requests
import logging
from tempfile import TemporaryDirectory
from unittest.mock import patch

from recommenders.datasets.download_utils import (
    maybe_download,
    download_path,
    is_valid_zip,
)


@pytest.fixture
def files_fixtures():
    file_url = (
        "https://raw.githubusercontent.com/recommenders-team/recommenders/main/LICENSE"
    )
    filepath = "license.txt"
    return file_url, filepath


def test_maybe_download(files_fixtures):
    file_url, filepath = files_fixtures
    if os.path.exists(filepath):
        os.remove(filepath)

    downloaded_filepath = maybe_download(file_url, "license.txt", expected_bytes=1212)
    assert os.path.exists(downloaded_filepath) is True
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
    assert os.path.exists(downloaded_filepath) is True
    maybe_download(file_url, "license.txt")
    assert "File ." + os.path.sep + "license.txt already downloaded" in caplog.text


def test_maybe_download_retry(caplog):
    caplog.clear()
    caplog.set_level(logging.INFO)
    with pytest.raises(requests.exceptions.HTTPError):
        maybe_download(
            "https://raw.githubusercontent.com/recommenders-team/resources/main/non_existing_file.zip"
        )
        assert "Problem downloading" in caplog.text


def test_maybe_download_sets_default_timeout(tmp_path, monkeypatch):
    captured = {}

    class Response:
        status_code = 200
        headers = {"content-length": "4"}

        def iter_content(self, block_size):
            yield b"done"

    def mock_get(url, stream, timeout):
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr("recommenders.datasets.download_utils.requests.get", mock_get)

    downloaded = maybe_download(
        "https://example.com/file.txt",
        "file.txt",
        work_directory=tmp_path,
    )

    assert captured["timeout"] == (10, 60)
    assert os.path.exists(downloaded)


def test_is_valid_zip(tmp):
    valid_zip = os.path.join(tmp, "valid.zip")
    with zipfile.ZipFile(valid_zip, "w") as zf:
        zf.writestr("test.txt", "hello")
    assert is_valid_zip(valid_zip) is True

    corrupt_zip = os.path.join(tmp, "corrupt.zip")
    with open(corrupt_zip, "wb") as f:
        f.write(b"this is not a zip file")
    assert is_valid_zip(corrupt_zip) is False


def test_maybe_download_redownloads_corrupt_zip(tmp, caplog):
    caplog.clear()
    caplog.set_level(logging.WARNING)
    corrupt_zip = os.path.join(tmp, "ml-100k.zip")
    with open(corrupt_zip, "wb") as f:
        f.write(b"truncated content")
    assert os.path.exists(corrupt_zip)

    url = (
        "https://raw.githubusercontent.com/recommenders-team/recommenders/main/LICENSE"
    )
    downloaded = maybe_download(url, "ml-100k.zip", work_directory=tmp)
    assert "corrupt, re-downloading" in caplog.text
    assert os.path.exists(downloaded)
    assert os.path.getsize(downloaded) > 17


def test_download_path():
    # Check that the temporal path is created and deleted
    with download_path() as path:
        assert os.path.isdir(path) is True
    assert os.path.isdir(path) is False

    # Check the behavior when a path is provided
    tmp_dir = TemporaryDirectory()
    with download_path(tmp_dir.name) as path:
        assert os.path.isdir(path) is True
    assert os.path.isdir(path) is True


def test_maybe_download_num_attempts(tmp):
    url = "https://recommenders.invalid/does-not-exist.txt"
    with patch(
        "recommenders.datasets.download_utils._download",
        side_effect=requests.exceptions.ConnectionError("unreachable"),
    ) as mock_download:
        with pytest.raises(requests.exceptions.ConnectionError):
            maybe_download(
                url,
                "file.txt",
                work_directory=tmp,
                num_attempts=2,
                wait_random_min=1,
                wait_random_max=2,
            )

    assert mock_download.call_count == 2


def test_maybe_download_wait_between_attempts(tmp):
    url = "https://recommenders.invalid/does-not-exist.txt"
    with patch(
        "recommenders.datasets.download_utils._download",
        side_effect=requests.exceptions.ConnectionError("unreachable"),
    ):
        start = time.time()
        with pytest.raises(requests.exceptions.ConnectionError):
            maybe_download(
                url,
                "file.txt",
                work_directory=tmp,
                num_attempts=3,
                wait_random_min=200,
                wait_random_max=300,
            )
        elapsed = time.time() - start

    # Two waits of 200-300ms between the three attempts
    assert 0.4 <= elapsed < 1.0
