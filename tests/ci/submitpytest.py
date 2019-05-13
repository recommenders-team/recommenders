"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import sys
'''
print("This is the name of the script: ", sys.argv[0])
print("Number of arguments: ", len(sys.argv))
print("The arguments are: " , str(sys.argv))

pytest_str = sys.argv[1]
print("pytest_str ",pytest_str)
'''

from azureml.core import Workspace
import os, json, sys
import azureml.core
from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace
import azureml.core

# this is from https://aidemos.visualstudio.com/DevOps%20for%20AI%20-%20Demo/_git/DevOps-For-AI-AML-Demo?path=%2Faml_service%2F00-WorkSpace.py&version=GBmaster
# another good ref is https://github.com/Microsoft/MLAKSDeployAML/blob/master/00_AML_Configuration.ipynb
#
# Initialize a Workspace
#
print("SDK Version:", azureml.core.VERSION)
print('current dir is ' +os.curdir)
with open("tests/ci/config.json") as f:
    config = json.load(f)

workspace_name = config["workspace_name"]
resource_group = config["resource_group"]
subscription_id = config["subscription_id"]
location = config["location"]

print(" WS name ",workspace_name)
print("subscription_id ",subscription_id)

cli_auth = AzureCliAuthentication()

try:
    print("Trying to get ws")
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        auth=cli_auth,
    )

except:
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
# useful experiment, etc info https://docs.microsoft.com/en-us/python/api/overview/azure/ml/intro?view=azure-ml-py

#
# Create an Experiment
#
from azureml.core import Experiment
#experiment_name = 'train-on-amlcompute'
experiment_name = 'unit_tests_staging'
experiment = Experiment(workspace = ws, name = experiment_name)

#
# Check Available VM families
#
from azureml.core.compute import ComputeTarget, AmlCompute

AmlCompute.supported_vmsizes(workspace = ws)
#AmlCompute.supported_vmsizes(workspace = ws, location='southcentralus')

#
# Create a compute Resource
# from here https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml
#
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeTarget
import os

# choose a name for your cluster

compute_name = os.environ.get("AML_COMPUTE_CLUSTER_NAME", "cpucluster")

compute_min_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MIN_NODES", 0)
compute_max_nodes = os.environ.get("AML_COMPUTE_CLUSTER_MAX_NODES", 4)

# CPU VM = STANDARD_D2_V2. For using GPU VM, set SKU to STANDARD_NC6 
vm_size = os.environ.get("AML_COMPUTE_CLUSTER_SKU", "STANDARD_D2_V2")


if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and type(compute_target) is AmlCompute:
        print('found compute target. just use it. ' + compute_name)
else:
    print('creating a new compute target...')
    provisioning_config = AmlCompute.provisioning_configuration(vm_size = vm_size,
                                                                min_nodes = compute_min_nodes, 
                                                                max_nodes = compute_max_nodes)

    # create the cluster
    compute_target = ComputeTarget.create(ws, compute_name, provisioning_config)
    
    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it will use the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)
    
     # For a more detailed view of current AmlCompute status, use get_status()
    print(compute_target.get_status().serialize())

#
# Provision as a run based compute target
#
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.core.runconfig import DEFAULT_GPU_IMAGE

# create a new runconfig object
run_config = RunConfiguration()

# signal that you want to use AmlCompute to execute script.
# set to 'local' instead of amlcompute when running in local docker
run_config.target = "amlcompute"

# AmlCompute will be created in the same region as workspace
# Set vm size for AmlCompute
# chosen from STANDARD_D2_V2 or STANDARD_NC6

run_config.amlcompute.vm_size = 'STANDARD_D2_V2'


# Do NOT enable Docker 
run_config.environment.docker.enabled = True

# set Docker base image to the default CPU-based image

#run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
# just to see if this is it bz
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE

#run_config.environment.docker.base_image = 'continuumio/miniconda3'

# use conda_dependencies.yml to create a conda environment in the Docker image for execution
run_config.environment.python.user_managed_dependencies = False

# auto-prepare the Docker image when used for execution (if it is not already prepared)
run_config.auto_prepare_environment = True

import os

print("reco_base.yaml path")
print(os.path.dirname('./reco_base.yaml'))
print('reco_base.yaml exists ',os.path.exists('./reco_base.yaml'))

# specify CondaDependencies obj
run_config.environment.python.conda_dependencies = CondaDependencies(conda_dependencies_file_path='./reco_base.yaml')

print("before import ScriptRunConfig")

# Now submit a run on AmlCompute
from azureml.core.script_run_config import ScriptRunConfig

print("before folder = .")
project_folder = "."
print('before ScriptRunconfig')

script_run_config = ScriptRunConfig(source_directory=project_folder,
                                    script='./tests/ci/runpytest-template.py',
                                    run_config=run_config)

print('before submit')
run = experiment.submit(script_run_config)
print('after submit')
run.wait_for_completion(show_output=True, wait_post_processing=True)

# go to azure portal to see log in azure ws and look for experiment name and look for individual run
print('files',run.get_file_names())
run.download_files(prefix='reports')


