# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
run_groupwise_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""

import sys
import logging
import pytest
import argparse
import mlflow
from test_groups import nightly_test_groups, pr_gate_test_groups

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument(
        "--expname",
        action="store",
        default="persistentAzureML",
        help="Experiment name on AzureML",
    )
    parser.add_argument(
        "--testkind",
        action="store",
        default="unit",
        help="Test kind - nightly or unit",
    )
    parser.add_argument(
        "--testgroup",
        action="store",
        default="group_cpu_001",
        help="Group name for the tests",
    )
    parser.add_argument(
        "--disable-warnings",
        action="store_true",
        help="Turn off warnings",
    )
    return parser.parse_args()


if __name__ == "__main__":

    logger = logging.getLogger("run_groupwise_pytest.py")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    args = parse_args()
    if args.testkind == "nightly":
        test_group = nightly_test_groups[args.testgroup]
    else:
        test_group = pr_gate_test_groups[args.testgroup]

    # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-log-view-metrics?view=azureml-api-2&tabs=jobs
    mlflow.autolog()

    # Add options to pytest command (Duration and disable warnings)
    pytest_string = test_group + ["--durations"] + ["0"]
    if args.disable_warnings is True:
        pytest_string += ["--disable-warnings"]

    # Execute pytest command
    logger.info("Executing tests now...")
    pytest_exit_code = pytest.main(pytest_string)
    logger.info("Test execution completed!")

    # Log pytest exit code as a metric
    # to be used to indicate success/failure in GitHub workflow
    mlflow.log_metric("pytest_exit_code", pytest_exit_code.value)
