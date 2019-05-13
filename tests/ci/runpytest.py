import subprocess
from azureml.core import Run
print('before run.get_context')
run = Run.get_context()
print('before subprocess.run')
#subprocess.run(["python", "-m", "pytest","tests/unit","-m","not notebooks and not spark and not gpu", "--junitxml=reports/test-unit.xml"])
#subprocess.run(["pytest","tests/unit/test_python_utils.py","-m","not notebooks and not spark and gpu", "--junitxml=reports/test-unit.xml","-s", "|","./logs/pytest.log"])
#subprocess.run(["pytest","tests/unit/test_dataset.py","-m","not notebooks and not spark and not gpu", "--junitxml=reports/test-unit.xml"])
subprocess.run(["pytest","tests/unit","-m","not notebooks and not spark and gpu", "--junitxml=reports/test-unit.xml"])

import os
print("os.listdir files", os.listdir("."))
# set up reports
name_of_upload = "reports"
path_on_disk = "reports"
run.upload_folder(name_of_upload, path_on_disk)