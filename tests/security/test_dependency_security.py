# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import requests
import pandas as pd

try:
    import tensorflow as tf
except ImportError:
    pass  # skip this import if we are in cpu environment


def test_requests():
    # Security issue: https://github.com/psf/requests/releases/tag/v2.31.0
    assert requests.__version__ >= "2.31.0"


def test_pandas():
    # Security issue: https://github.com/advisories/GHSA-cmm9-mgm5-9r42
    assert pd.__version__ >= "1.0.3"


@pytest.mark.gpu
def test_tensorflow():
    assert tf.__version__ >= "2.6.0"
