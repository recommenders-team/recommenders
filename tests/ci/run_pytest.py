# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
run_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""

import argparse
import subprocess
import logging
import os

from azureml.core import Run


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process inputs')
    # test folder
    parser.add_argument("--testfolder", "-f",
                        action="store",
                        default="./tests/unit",
                        help="Folder where tests are located")
    parser.add_argument("--num",
                        action="store",
                        default="99",
                        help="test num")
    # test markers
    parser.add_argument("--testmarkers", "-m",
                        action="store",
                        default="not notebooks and not spark and not gpu",
                        help="Specify test markers for test selection")
    # test results file
    parser.add_argument("--xmlname", "-j",
                        action="store",
                        default="reports/test-unit.xml",
                        help="Test results")
    args = parser.parse_args()

    return args 


if __name__ == "__main__":
    logger = logging.getLogger("submit_azureml_pytest.py")
    args = create_arg_parser()

    logger.debug('junit_xml', args.xmlname)

    # Run.get_context() is needed to save context as pytest causes corruption
    # of env vars
    run = Run.get_context()
    '''
    This is an example of a working subprocess.run for a unit test run:
    subprocess.run(["pytest", "tests/unit",
                    "-m", "not notebooks and not spark and not gpu",
                    "--junitxml=reports/test-unit.xml"])
    '''
    logger.debug("args.junitxml", args.xmlname)
    logger.debug("junit=", "--junitxml="+args.xmlname)
    logger.info('pytest run:',
                ["pytest",
                 args.testfolder,
                 "-m",
                 args.testmarkers,
                 "--junitxml="+args.xmlname])
    subprocess.run(["pytest",
                    args.testfolder,
                    "-m",
                    args.testmarkers,
                    "--junitxml="+args.xmlname])
    #
    # Leveraged code from this  notebook:
    # https://msdata.visualstudio.com/Vienna/_search?action=contents&text=upload_folder&type=code&lp=code-Project&filters=ProjectFilters%7BVienna%7DRepositoryFilters%7BAzureMlCli%7D&pageSize=25&sortOptions=%5B%7B%22field%22%3A%22relevance%22%2C%22sortOrder%22%3A%22desc%22%7D%5D&result=DefaultCollection%2FVienna%2FAzureMlCli%2FGBmaster%2F%2Fsrc%2Fazureml-core%2Fazureml%2Fcore%2Frun.py
    logger.debug("os.listdir files", os.listdir("."))
    logger.debug("os.listdir reports", os.listdir("./reports"))

    #  files for AzureML
    name_of_upload = "reports"
    path_on_disk = "./reports"
    run.upload_folder(name_of_upload, path_on_disk)
