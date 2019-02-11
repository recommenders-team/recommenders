# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
import pytest
from reco_utils.dataset.url_utils import maybe_download


def test_maybe_download():
    file_url = "https://raw.githubusercontent.com/Microsoft/Recommenders/master/LICENSE"
    filepath = "license.txt"
    assert not os.path.exists(filepath)
    filepath = maybe_download(file_url, "license.txt", expected_bytes=1162)
    assert os.path.exists(filepath)
    os.remove(filepath)
    with pytest.raises(IOError):
        filepath = maybe_download(file_url, "license.txt", expected_bytes=0)
