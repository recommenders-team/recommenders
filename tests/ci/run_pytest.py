# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
'''
run_pytest.py is the script submitted to Azure ML that runs pytest
'''

import subprocess
import os

from azureml.core import Run
print('before run.get_context')
run = Run.get_context()
print('before subprocess.run')
# start_time = time.time
subprocess.run(["pytest", "tests/unit",
                "-m", "not notebooks and not spark and not gpu",
                "--junitxml=reports/test-unit.xml"])
# run_time = time.time - start_time
# print("pytest run time ", time.localtime(run_time))
# run.tag('pytest run time ', time.localtime(run_time))

print("os.listdir files", os.listdir("."))
# set up reports
name_of_upload = "reports"
path_on_disk = "reports"
run.upload_folder(name_of_upload, path_on_disk)
run.upload_file("outputs", "reports/test-unit.xml")
