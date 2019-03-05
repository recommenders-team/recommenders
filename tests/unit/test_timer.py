# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pytest
from reco_utils.common.timer import Timer


@pytest.fixture(scope="function")
def t():
    return Timer()


def test_no_time(t):
    assert t.interval == 0
    with Timer() as t2:
        assert t2.interval == 0


def test_stop_before_start(t):
    with pytest.raises(ValueError):
        t.stop()


def test_timer(t):
    big_num = 1000
    t.start()
    r = 0
    a = [r + i for i in range(big_num)]
    t.stop()
    assert t.interval < 1
    r = 0
    with Timer() as t2:
        a = [r + i for i in range(big_num)]
    assert t2.interval < 1


def test_timer_format(t):
    assert str(t) == "0:00:00"
    assert str(t.interval) == "0"
