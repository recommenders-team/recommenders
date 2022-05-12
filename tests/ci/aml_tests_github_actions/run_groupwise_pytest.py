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
import glob
from test_groups import nightly_test_groups, unit_test_groups
from time import time

if __name__ == "__main__":

    logger = logging.getLogger("submit_groupwise_azureml_pytest.py")
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    parser = argparse.ArgumentParser(description="Process inputs")
    parser.add_argument(
        "--testkind",
        "-k",
        action="store",
        default="unit",
        help="Test kind - nightly or unit",
    )
    parser.add_argument(
        "--testgroup",
        "-g",
        action="store",
        default="group_cpu_001",
        help="Group name for the tests",
    )
    args = parser.parse_args()

    # if args.testkind == "nightly":
    #     test_group = nightly_test_groups[args.testgroup]
    # else:
    #     test_group = unit_test_groups[args.testgroup]
    test_group = ["tests/integration/examples/test_notebooks_gpu.py::test_sasrec_quickstart_integration"]

    logger.info("Tests to be executed")
    logger.info(str(test_group))

    # Run.get_context() is needed to save context as pytest causes corruption
    # of env vars
    run = Run.get_context()

    logger.info("Python version ")
    logger.info(str(sys.version))
    logger.info("Executing tests now...")

    start_time = time()
    # execute pytest command
    pytest_exit_code = pytest.main(test_group)
    print("time taken for execute pytest command ", time()-start_time)
    
    logger.info("Test execution completed!")

    # log pytest exit code as a metric
    # to be used to indicate success/failure in github workflow
    run.log("pytest_exit_code", pytest_exit_code.value)

    # #
    # # Leveraged code from this  notebook:
    # # https://msdata.visualstudio.com/Vienna/_search?action=contents&text=upload_folder&type=code&lp=code-Project&filters=ProjectFilters%7BVienna%7DRepositoryFilters%7BAzureMlCli%7D&pageSize=25&sortOptions=%5B%7B%22field%22%3A%22relevance%22%2C%22sortOrder%22%3A%22desc%22%7D%5D&result=DefaultCollection%2FVienna%2FAzureMlCli%2FGBmaster%2F%2Fsrc%2Fazureml-core%2Fazureml%2Fcore%2Frun.py
    # logger.info("os.listdir files {}".format(os.listdir(".")))

    # upload pytest stdout file
    logs_path = glob.glob('**/70_driver_log.txt', recursive=True)[0]
    run.upload_file(name='test_logs', path_or_stream=logs_path)
    run.upload_file(name='output.ipynb', path_or_stream="output.ipynb")
