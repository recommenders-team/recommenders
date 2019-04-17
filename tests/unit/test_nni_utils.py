# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import pytest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from reco_utils.nni.nni_utils import get_experiment_status, check_experiment_status, check_stopped, \
    check_metrics_written, get_trials, NNI_STATUS_URL


class MockResponse:
    # Class that mocks requests.models.Response
    def __init__(self, content):
        self._content = content

    def json(self):
        return self._content


def mocked_requests_get(url, content):
    return MockResponse(content)


def mock_exception():
    raise Exception()


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'some_status'}))
def test_get_experiment_status(unused_class):
    assert 'some_status' == get_experiment_status(NNI_STATUS_URL)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'DONE'}))
def test_check_experiment_status_done(unused_class):
    check_experiment_status(1)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'TUNER_NO_MORE_TRIAL'}))
def test_check_experiment_status_tuner_no_more_trial(unused_class):
    check_experiment_status(1)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'RUNNING'}))
def test_check_experiment_status_running(unused_class):
    with pytest.raises(TimeoutError) as excinfo:
        check_experiment_status(1)
    assert "check_experiment_status() timed out" == str(excinfo.value)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'NO_MORE_TRIAL'}))
def test_check_experiment_status_no_more_trial(unused_class):
    with pytest.raises(TimeoutError) as excinfo:
        check_experiment_status(1)
    assert "check_experiment_status() timed out" == str(excinfo.value)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'some_failed_status'}))
def test_check_experiment_status_failed(unused_class):
    with pytest.raises(RuntimeError) as excinfo:
        check_experiment_status(1)
    assert "NNI experiment failed to complete with status some_failed_status" == str(excinfo.value)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, {'status': 'some_status'}))
def test_check_stopped_timeout(unused_class):
    with pytest.raises(TimeoutError) as excinfo:
        check_stopped(wait=1, max_retries=1)
    assert "check_stopped() timed out" == str(excinfo.value)


@patch('requests.get', side_effect=mock_exception)
def test_check_stopped(unused_class):
    check_stopped(wait=1, max_retries=1)


@patch('requests.get',
       side_effect=lambda url: mocked_requests_get(url, [{'finalMetricData': None}, {'finalMetricData': None}]))
def test_check_metrics_written(unused_class):
    check_metrics_written(wait=1, max_retries=1)


@patch('requests.get', side_effect=lambda url: mocked_requests_get(url, [{'logPath': '/p'}, {'logPath': '/q'}]))
def test_check_metrics_written_timeout(unused_class):
    with pytest.raises(TimeoutError) as excinfo:
        check_metrics_written(wait=1, max_retries=1)
    assert "check_metrics_written() timed out" == str(excinfo.value)
