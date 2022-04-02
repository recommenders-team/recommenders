# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
run_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""

import logging
import os
import sys
from azureml.core import Run
import pytest
import json
import argparse


if __name__ == "__main__":

    logger = logging.getLogger("submit_azureml_pytest.py")

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # logs_path = "user_logs/std_log.txt"
    # logs_path = "azureml-logs/70_driver_log.txt"

    logs_path = "pytest_logs.txt"
    # logging.basicConfig(filename=logs_path, level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument(
        "--testgroup",
        "-g",
        action="store",
        default="group_criteo",
        help="Group name for the tests",
    )
    args = parser.parse_args()

    with open("tests/ci/aml_tests_github_actions/test_module_groups.json") as f:
        test_group = json.load(f)[args.testgroup]
    
    logger.info("Tests to be executed")
    logger.info(str(test_group))

    # Run.get_context() is needed to save context as pytest causes corruption
    # of env vars
    run = Run.get_context()
    """
    This is an example of a working subprocess.run for a unit test run:
    subprocess.run(["pytest", "tests/unit",
                    "-m", "not notebooks and not spark and not gpu",
                    "--junitxml=reports/test-unit.xml"])
    """

    logger.info("Python version ")
    logger.info(str(sys.version))
    logger.info("Executing tests now...")

    # execute pytest command
    pytest_exit_code = pytest.main(test_group + ["-o log_cli=true", "--log-file", logs_path, "--log-file-level", "INFO"])
    
    logger.info("Test execution completed!")

    # log pytest exit code as a metric
    # to be used to indicate success/failure in github workflow
    run.log("pytest_exit_code", pytest_exit_code.value)

    #
    # Leveraged code from this  notebook:
    # https://msdata.visualstudio.com/Vienna/_search?action=contents&text=upload_folder&type=code&lp=code-Project&filters=ProjectFilters%7BVienna%7DRepositoryFilters%7BAzureMlCli%7D&pageSize=25&sortOptions=%5B%7B%22field%22%3A%22relevance%22%2C%22sortOrder%22%3A%22desc%22%7D%5D&result=DefaultCollection%2FVienna%2FAzureMlCli%2FGBmaster%2F%2Fsrc%2Fazureml-core%2Fazureml%2Fcore%2Frun.py
    logger.debug("os.listdir files {}".format(os.listdir(".")))

    # #  files for AzureML
    # name_of_upload = "reports"
    # path_on_disk = "./reports"
    # run.upload_folder(name_of_upload, path_on_disk)

    # upload pytest stdout file
    run.upload_file(name='test_logs', path_or_stream=logs_path)
