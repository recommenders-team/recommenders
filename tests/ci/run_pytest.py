# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
run_pytest.py is the script submitted to Azure ML that runs pytest.
pytest runs all tests in the specified test folder unless parameters
are set otherwise.
"""

import argparse
import subprocess

from azureml.core import Run


def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process inputs')
    # test folder
    parser.add_argument("--testfolder", "-f",
                        action="store",
                        default="./tests/unit",
                        help="Folder where tests are located")
    # internal test purposes
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
    parser.add_argument("--junitxml", "-j",
                        action="store",
                        default="reports/test-unit.xml",
                        help="Test results")

    args = parser.parse_args()
    return(args)


def run_pytest(test_folder="./tests/unit",
               test_markers="not notebooks and not spark and not gpu",
               junitxml="--junitxml=reports/test-unit.xml"):
    '''
    This is the script that is submitted to AzureML to run pytest.

    Args:
         test_folder  (str): folder that contains the tests that pytest runs
         test_markers (str): test markers used by pytest "not notebooks and
                             not spark and not gpu"
         junitxml     (str): file of output summary of tests run
                             note "--junitxml" is required as part of
                             the string
                                Example: "--junitxml=reports/test-unit.xml"
    Return: none

    '''
    # Run.get_context() is needed to save context as pytest causes corruption
    # of env vars
    run = Run.get_context()
    '''
    This is an example of a working subprocess.run for a unit test run:
    subprocess.run(["pytest", "tests/unit",
                    "-m", "not notebooks and not spark and not gpu",
                    "--junitxml=reports/test-unit.xml"])
    '''

    print('pytest run:', ["pytest", test_folder, "-m", test_markers, junitxml])
    subprocess.run(["pytest", test_folder, "-m", test_markers, junitxml])

    #  files for AzureML
    name_of_upload = "reports"
    path_on_disk = "./reports"
    run.upload_folder(name_of_upload, path_on_disk)

    # logger.debug(("os.listdir files", os.listdir("."))
    # logger.debug(("os.listdir reports", os.listdir("./reports"))
    # logger.debug(("os.listdir outputs", os.listdir("./outputs"))

    # Leveraged code from this  notebook:
    # https://msdata.visualstudio.com/Vienna/_search?action=contents&text=upload_folder&type=code&lp=code-Project&filters=ProjectFilters%7BVienna%7DRepositoryFilters%7BAzureMlCli%7D&pageSize=25&sortOptions=%5B%7B%22field%22%3A%22relevance%22%2C%22sortOrder%22%3A%22desc%22%7D%5D&result=DefaultCollection%2FVienna%2FAzureMlCli%2FGBmaster%2F%2Fsrc%2Fazureml-core%2Fazureml%2Fcore%2Frun.py


if __name__ == "__main__":

    args = create_arg_parser()

    # run_pytest()
    junit_str = "--junitxml="+args.junitxml
    # logger.debug(('junit_str', junit_str)
    run_pytest(test_folder=args.testfolder,
               test_markers=args.testmarkers,
               junitxml=junit_str)
