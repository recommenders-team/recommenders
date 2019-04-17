# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch
import pytest

from reco_utils.nni.nni_utils import (
    get_experiment_status,
    check_experiment_status,
    check_stopped,
    check_metrics_written,
    get_trials,
    NNI_STATUS_URL,
    NNI_TRIAL_JOBS_URL
)


class MockResponse:
    # Class that mocks requests.models.Response
    def __init__(self, content):
        self._content = content

    def json(self):
        return self._content


def mocked_status_get(url, content):
    assert url.startswith(NNI_STATUS_URL)
    return MockResponse(content)


def mocked_trials_get(url, content):
    assert url.startswith(NNI_TRIAL_JOBS_URL)
    return MockResponse(content)
    

@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'some_status'}))
def test_get_experiment_status():
    assert 'some_status' == get_experiment_status(NNI_STATUS_URL)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'DONE'}))
def test_check_experiment_status_done():
    check_experiment_status(1)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'TUNER_NO_MORE_TRIAL'}))
def test_check_experiment_status_tuner_no_more_trial():
    check_experiment_status(1)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'RUNNING'}))
def test_check_experiment_status_running():
    with pytest.raises(TimeoutError) as excinfo:
        check_experiment_status(1)
    assert "check_experiment_status() timed out" == str(excinfo.value)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'NO_MORE_TRIAL'}))
def test_check_experiment_status_no_more_trial():
    with pytest.raises(TimeoutError) as excinfo:
        check_experiment_status(1)
    assert "check_experiment_status() timed out" == str(excinfo.value)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'some_failed_status'}))
def test_check_experiment_status_failed():
    with pytest.raises(RuntimeError) as excinfo:
        check_experiment_status(1)
    assert "NNI experiment failed to complete with status some_failed_status" == str(excinfo.value)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, {'status': 'some_status'}))
def test_check_stopped_timeout():
    with pytest.raises(TimeoutError) as excinfo:
        check_stopped(wait=1, max_retries=1)
    assert "check_stopped() timed out" == str(excinfo.value)


def test_check_stopped():
    check_stopped(wait=1, max_retries=1)


@patch('requests.get',
       side_effect=lambda url: mocked_status_get(url, [{'finalMetricData': None}, {'finalMetricData': None}]))
def test_check_metrics_written():
    check_metrics_written(wait=1, max_retries=1)


@patch('requests.get', side_effect=lambda url: mocked_status_get(url, [{'logPath': '/p'}, {'logPath': '/q'}]))
def test_check_metrics_written_timeout():
    with pytest.raises(TimeoutError) as excinfo:
        check_metrics_written(wait=1, max_retries=1)
    assert "check_metrics_written() timed out" == str(excinfo.value)


def test_get_trials():
    with TemporaryDirectory() as tmp_dir1, TemporaryDirectory() as tmp_dir2:
        mock_trials = [
            {'finalMetricData': [{'data': '{"rmse":0.8,"default":0.3}'}], 
            'logPath': 'file://localhost:{}'.format(tmp_dir1)},
            {'finalMetricData': [{'data': '{"rmse":0.9,"default":0.2}'}], 
            'logPath': 'file://localhost:{}'.format(tmp_dir2)},
        ]
        metrics1 = {"rmse": 0.8, precision_at_k": 0.3}
        with open(os.path.join(tmp_dir1, 'metrics.json'), 'w') as f:
            json.dump(metrics1, f)
        params1 = {"parameter_id": 1, "parameter_source": "algorithm", 
        "parameters": {"n_factors": 100, "reg": 0.1}}
        with open(os.path.join(tmp_dir1, 'parameter.cfg'), 'w') as f:
            json.dump(params1, f)
        metrics2 = {"rmse": 0.9, precision_at_k": 0.2}
        with open(os.path.join(tmp_dir2, 'metrics.json'), 'w') as f:
            json.dump(metrics2, f)
        params2 = {"parameter_id": 2, "parameter_source": "algorithm", 
        "parameters": {"n_factors": 50, "reg": 0.02}}
        with open(os.path.join(tmp_dir2, 'parameter.cfg'), 'w') as f:
            json.dump(params2, f)

        with patch('requests.get', side_effect=lambda url: mocked_trials_get(url, mock_trials)):
            trials, best_metrics, best_params, best_trial_path = get_trials(optimize_mode='maximize')

        assert trials == [({'default': 1}, tmp_dir), ({'default': 2}, tmp_dir)]
        assert best_metrics == {'a': 1}
        assert best_params == {'b': 2}
        assert best_trial_path == tmp_dir
