# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import re
import logging
import pytest
import requests
from unittest.mock import patch

from recommenders.datasets.movielens import (
    MOVIELENS_URL,
    MOVIELENS_BACKUP_URL,
    ERROR_MOVIE_LENS_SIZE,
    download_movielens,
)


@pytest.mark.parametrize("size", ["100k", "1m", "10m", "20m"])
def test_download_movielens_uses_original_url(size, tmp):
    with patch("recommenders.datasets.movielens.maybe_download") as mock_download:
        download_movielens(size, dest_path=f"{tmp}/ml.zip")

    mock_download.assert_called_once_with(
        MOVIELENS_URL.format(size=size), "ml.zip", work_directory=tmp, num_attempts=3
    )


@pytest.mark.parametrize("size", ["100k", "1m", "10m", "20m"])
def test_download_movielens_falls_back_to_backup(size, tmp, caplog):
    caplog.clear()
    caplog.set_level(logging.WARNING)

    with patch(
        "recommenders.datasets.movielens.maybe_download",
        side_effect=[requests.exceptions.SSLError("expired certificate"), None],
    ) as mock_download:
        download_movielens(size, dest_path=f"{tmp}/ml.zip")

    assert mock_download.call_count == 2
    assert mock_download.call_args_list[0].args[0] == MOVIELENS_URL.format(size=size)
    assert mock_download.call_args_list[1].args[0] == MOVIELENS_BACKUP_URL.format(
        size=size
    )
    assert MOVIELENS_BACKUP_URL.format(size=size) in caplog.text
    # The original URL fails fast, the backup uses the default number of attempts
    assert mock_download.call_args_list[0].kwargs["num_attempts"] == 3
    assert "num_attempts" not in mock_download.call_args_list[1].kwargs


def test_download_movielens_raises_when_backup_also_fails(tmp):
    with patch(
        "recommenders.datasets.movielens.maybe_download",
        side_effect=requests.exceptions.ConnectionError("unreachable"),
    ):
        with pytest.raises(requests.exceptions.ConnectionError):
            download_movielens("100k", dest_path=f"{tmp}/ml.zip")


def test_download_movielens_invalid_size(tmp):
    with pytest.raises(ValueError, match=re.escape(ERROR_MOVIE_LENS_SIZE)):
        download_movielens("bad_size", dest_path=f"{tmp}/ml.zip")
