# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import json
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
# uncomment if using gpu
# from azureml.core.runconfig import DEFAULT_GPU_IMAGE
from azureml.core.script_run_config import ScriptRunConfig

# this is from
# https://aidemos.visualstudio.com/DevOps%20for%20AI%20-%20Demo/_git/DevOps-For-AI-AML-Demo?path=%2Faml_service%2F00-WorkSpace.py&version=GBmaster
# another good ref is
# https://github.com/Microsoft/MLAKSDeployAML/blob/master/00_AML_Configuration.ipynb
#
# Initialize a Workspace

print("SDK Version:", azureml.core.VERSION)
print('current dir is ' + os.curdir)
with open("tests/ci/config.json") as f:
    config = json.load(f)

workspace_name = config["workspace_name"]
resource_group = config["resource_group"]
subscription_id = config["subscription_id"]
location = config["location"]

print(" WS name ", workspace_name)
print("subscription_id ", subscription_id)

cli_auth = AzureCliAuthentication()

try:
    print("Trying to get ws")
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth,
    )

except Exception:
    # this call might take a minute or two.
    print("Creating new workspace")
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        # create_resource_group=True,
        location=location,
        auth=cli_auth,
    )

# print Workspace details
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep="\n")
#
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training/train-on-amlcompute/train-on-amlcompute.ipynb
# useful experiment, etc info
# https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py

#
# Create an Experiment
#

# experiment_name = 'train-on-amlcompute'
experiment_name = 'unit_tests_staging'
experiment = Experiment(workspace=ws, name=experiment_name)

#
# Check Available VM families
#
'''
bz
AmlCompute.supported_vmsizes(workspace=ws)
# AmlCompute.supported_vmsizes(workspace = ws, location='southcentralus')

#
# Create a compute Resource
# from here
# https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml
# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets
#
'''

# create a new runconfig object
run_config = RunConfiguration()

# signal that you want to use AmlCompute to execute script.
# set to 'local' instead of amlcompute when running in local docker
run_config.target = "amlcompute"

# AmlCompute will be created in the same region as workspace
# Set vm size for AmlCompute
# chosen from STANDARD_D2_V2 or STANDARD_NC6

run_config.amlcompute.vm_size = 'STANDARD_D2_V2'


# enable Docker
run_config.environment.docker.enabled = True

# set Docker base image to the default CPU-based image

# run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
# just to see if this is it bz
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

# run_config.environment.docker.base_image = 'continuumio/miniconda3'

# use conda_dependencies.yml to create a conda environment in the
# Docker image for execution
run_config.environment.python.user_managed_dependencies = False

# auto-prepare the Docker image when used for execution (if it is not already
# prepared)
run_config.auto_prepare_environment = True

print("reco_base.yaml path")
print(os.path.dirname('./reco_base.yaml'))
print('reco_base.yaml exists ', os.path.exists('./reco_base.yaml'))

# specify CondaDependencies obj
run_config.environment.python.conda_dependencies = CondaDependencies(
    conda_dependencies_file_path='./reco_base.yaml')

print("before import ScriptRunConfig")

# Now submit a run on AmlCompute

print("before folder = .")
project_folder = "."
print('before ScriptRunconfig')

script_run_config = ScriptRunConfig(source_directory=project_folder,
                                    script='./tests/ci/runpytest.py',
                                    run_config=run_config)

print('before submit')
run = experiment.submit(script_run_config)
print('after submit')
run.wait_for_completion(show_output=True, wait_post_processing=True)

# go to azure portal to see log in azure ws and look for experiment name and
# look for individual run
print('files', run.get_file_names())
run.download_files(prefix='reports')
