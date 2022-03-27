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


if __name__ == "__main__":
    logger = logging.getLogger("submit_azureml_pytest.py")

    test_group = [
        "tests/smoke/recommenders/dataset/test_criteo.py::test_criteo_load_pandas_df",
        "tests/integration/recommenders/datasets/test_criteo.py::test_criteo_load_pandas_df",
    ]

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
    pytest_exit_code = pytest.main(test_group)
    
    logger.info("Test execution completed!")

    # log pytest exit code as a metric
    # to be used to indicate success/failure in github workflow
    run.log("pytest_exit_code", pytest_exit_code.value)

    #
    # Leveraged code from this  notebook:
    # https://msdata.visualstudio.com/Vienna/_search?action=contents&text=upload_folder&type=code&lp=code-Project&filters=ProjectFilters%7BVienna%7DRepositoryFilters%7BAzureMlCli%7D&pageSize=25&sortOptions=%5B%7B%22field%22%3A%22relevance%22%2C%22sortOrder%22%3A%22desc%22%7D%5D&result=DefaultCollection%2FVienna%2FAzureMlCli%2FGBmaster%2F%2Fsrc%2Fazureml-core%2Fazureml%2Fcore%2Frun.py
    logger.debug("os.listdir files {}".format(os.listdir(".")))

    #  files for AzureML
    name_of_upload = "reports"
    path_on_disk = "./reports"
    run.upload_folder(name_of_upload, path_on_disk)

    # upload pytest stdout file
    run.upload_file(name='test_logs', path_or_stream="user_logs/std_log.txt")
