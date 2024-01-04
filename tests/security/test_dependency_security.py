# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest
import requests
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    import torch
except ImportError:
    pass  # skip this import if we are in cpu environment


def test_requests():
    # Security issue: https://github.com/psf/requests/releases/tag/v2.31.0
    assert requests.__version__ >= "2.31.0"


def test_numpy():
    # Security issue: https://github.com/advisories/GHSA-frgw-fgh6-9g52
    assert np.__version__ >= "1.13.3"


def test_pandas():
    # Security issue: https://github.com/advisories/GHSA-cmm9-mgm5-9r42
    assert pd.__version__ >= "1.0.3"


@pytest.mark.gpu
def test_tensorflow():
    # Security issue: https://github.com/advisories/GHSA-w5gh-2wr2-pm6g
    # Security issue: https://github.com/advisories/GHSA-r6jx-9g48-2r5r
    # Security issue: https://github.com/advisories/GHSA-xxcj-rhqg-m46g
    assert tf.__version__ >= "2.8.4"


@pytest.mark.gpu
def test_torch():
    # Security issue: https://github.com/advisories/GHSA-47fc-vmwq-366v
    assert torch.__version__ >= "1.13.1"
