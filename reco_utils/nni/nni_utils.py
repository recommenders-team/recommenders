import json
import numpy as np
import os
import requests
import time

NNI_REST_ENDPOINT = 'http://localhost:8080/api/v1/nni'
NNI_STATUS_URL = NNI_REST_ENDPOINT + '/check-status'
NNI_TRIAL_JOBS_URL = NNI_REST_ENDPOINT + '/trial-jobs'


def get_experiment_status(status_url):
    nni_status = requests.get(status_url).json()
    return nni_status['status']


def check_experiment_status():
    while True:
        time.sleep(20)
        status = get_experiment_status(NNI_STATUS_URL)
        if status == 'DONE' or status == 'NO_MORE_TRIAL':
            break
        elif status != 'RUNNING':
            raise RuntimeError("NNI experiment failed to complete with status {}".format(status))


def check_stopped():
    while True:
        time.sleep(20)
        try:
            get_experiment_status(NNI_STATUS_URL)
        except:
            break


def get_trials(optimize_mode):
    if optimize_mode not in ['minimize', 'maximize']:
        raise ValueError("optimize_mode should equal either 'minimize' or 'maximize'")
    all_trials = requests.get(NNI_TRIAL_JOBS_URL).json()
    trials = [(eval(trial['finalMetricData'][0]['data']), trial['logPath'].split(':')[-1]) for trial in all_trials]
    optimize_fn = np.argmax if optimize_mode == 'maximize' else np.argmin
    ind_best = optimize_fn([trial[0]['default'] for trial in trials])
    best_trial = trials[ind_best]
    # Read the metrics from the trial directory in order to get the name of the default metric
    with open(os.path.join(best_trial[1], "metrics.json"), "r") as fp:
            best_metrics = json.load(fp)
    with open(os.path.join(best_trial[1], "parameter.cfg"), "r") as fp:
            best_params = json.load(fp)
    best_trial_path = best_trial[1]
    return trials, best_metrics, best_params, best_trial_path
