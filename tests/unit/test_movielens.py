# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.dataset import movielens


def test_movielens():
    size_100k = len(movielens.load_data())
    assert size_100k == 100000
    size_1m = len(movielens.load_data(size="1m"))
    assert size_1m == 1000209
    size_10m = len(movielens.load_data(size="10m"))
    assert size_10m == 10000054
    size_20m = len(movielens.load_data(size="20m"))
    assert size_20m == 20000263

    with pytest.raises(ValueError):
        movielens.load_data(size='10k')

    with pytest.raises(ValueError):
        movielens.load_data(header=['a', 'b', 'c'])
