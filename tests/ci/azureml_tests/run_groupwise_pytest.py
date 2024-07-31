# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
run_groupwise_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""

import argparse
import logging
import pytest
import sys

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
    return parser.parse_args()


if __name__ == "__main__":

    logger = logging.getLogger("run_groupwise_pytest.py")

    args = parse_args()
    if args.testkind == "nightly":
        test_group = nightly_test_groups[args.testgroup]
    else:
        test_group = pr_gate_test_groups[args.testgroup]

    # Add options to pytest command (Duration)
    pytest_string = test_group + ["--durations"] + ["0"]

    # Execute pytest command
    logger.info("Executing tests now...")
    sys.exit(pytest.main(pytest_string))
