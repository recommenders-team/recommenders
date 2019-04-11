# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pytest
import time
from reco_utils.common.timer import Timer


TOL = 0.01


@pytest.fixture(scope="function")
def t():
    return Timer()


def test_no_time(t):
    assert t.interval == 0
    assert t.running == False


def test_stop_before_start(t):
    with pytest.raises(ValueError):
        t.stop()


def test_interval_before_stop(t):
    t.start()
    with pytest.raises(ValueError):
        t.interval


def test_timer(t):
    t.start()
    assert t.running == True
    time.sleep(1)
    t.stop()
    assert t.running == False
    assert t.interval == pytest.approx(1, abs=TOL)
    with Timer() as t2:
        assert t2.running == True
        time.sleep(1)
    assert t2.interval == pytest.approx(1, abs=TOL)
    assert t2.running == False


def test_timer_format(t):
    assert str(t) == "0.0000"
    assert str(t.interval) == "0"
