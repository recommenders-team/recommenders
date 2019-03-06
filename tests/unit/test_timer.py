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
    with Timer() as t2:
        assert t2.interval == 0


def test_stop_before_start(t):
    with pytest.raises(ValueError):
        t.stop()


def test_timer(t):
    t.start()
    time.sleep(1)
    t.stop()
    assert t.interval == pytest.approx(1, abs=TOL)
    with Timer() as t2:
        time.sleep(1)
    assert t2.interval == pytest.approx(1, abs=TOL)


def test_timer_format(t):
    assert str(t) == "0:00:00"
    assert str(t.interval) == "0"
