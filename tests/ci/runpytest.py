# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import os

from azureml.core import Run
print('before run.get_context')
run = Run.get_context()
print('before subprocess.run')

subprocess.run(["pytest", "tests/unit/test_timer.py",
                "-m", "not notebooks and not spark and not gpu",
                "--junitxml=reports/test-unit.xml"])

print("os.listdir files", os.listdir("."))
# set up reports
name_of_upload = "reports"
path_on_disk = "reports"
run.upload_folder(name_of_upload, path_on_disk)
