# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import pytest
from tempfile import TemporaryDirectory
from reco_utils.dataset.download_utils import maybe_download, download_path


def test_maybe_download():
    file_url = "https://raw.githubusercontent.com/Microsoft/Recommenders/master/LICENSE"
    filepath = "license.txt"
    assert not os.path.exists(filepath)
    filepath = maybe_download(file_url, "license.txt", expected_bytes=1162)
    assert os.path.exists(filepath)
    os.remove(filepath)
    with pytest.raises(IOError):
        filepath = maybe_download(file_url, "license.txt", expected_bytes=0)


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
