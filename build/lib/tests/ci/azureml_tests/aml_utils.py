# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
This module includes utilities for tests on AzureML via AML Python SDK v2.
See
* https://learn.microsoft.com/en-us/azure/machine-learning/concept-v2?view=azureml-api-2
* https://learn.microsoft.com/en-us/azure/machine-learning/reference-migrate-sdk-v1-mlflow-tracking?view=azureml-api-2&tabs=aml%2Ccli%2Cmlflow
"""
import pathlib
import re

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute, BuildContext, Environment, Workspace
from azure.ai.ml.exceptions import JobException
from azure.core.exceptions import ResourceExistsError
from azure.identity import DefaultAzureCredential


def get_client(subscription_id, resource_group, workspace_name):
    """
    Get the client with specified AzureML workspace, or create one if not existing.
    See https://github.com/Azure/azureml-examples/blob/main/sdk/python/resources/workspace/workspace.ipynb
    """
    params = dict(
        credential=DefaultAzureCredential(),
        subscription_id=subscription_id,
        resource_group_name=resource_group,
    )
    client = MLClient(**params)

    workspace = client.workspaces.get(workspace_name)
    if workspace is None:
        workspace = client.workspaces.begin_create(
            Workspace(name=workspace_name)
        ).result()

    params["workspace_name"] = workspace_name
    client = MLClient(**params)
    return client


def create_or_start_compute(client, name, size, max_instances):
    """
    Start the specified compute.
    See https://github.com/Azure/azureml-examples/blob/main/sdk/python/resources/compute/compute.ipynb
    """
    compute = client.compute.get(name)
    if compute is None:
        compute = client.compute.begin_create_or_update(
            AmlCompute(
                name=name,
                type="amlcompute",
                size=size,
                max_instances=max_instances,
            )
        ).result()


def get_or_create_environment(
    client,
    environment_name,
    use_gpu,
    use_spark,
    conda_openjdk_version,
    python_version,
):
    """
    AzureML requires the run environment to be setup prior to submission.
    This configures a docker persistent compute.
    See https://github.com/Azure/azureml-examples/blob/main/sdk/python/assets/environment/environment.ipynb

    Args:
        client (MLClient): the client to interact with AzureML services
        environment_name (str): Environment name
        use_gpu (bool): True if gpu packages should be
            added to the conda environment, else False
        use_spark (bool): True if PySpark packages should be
            added to the conda environment, else False
        conda_openjdk_version (str): "21" by default
        python_version (str): python version, such as "3.11"
    """
    compute = "gpu" if use_gpu else "cpu"
    extras = (
        "[dev" + (",gpu" if use_gpu else "") + (",spark" if use_spark else "") + "]"
    )
    dockerfile = pathlib.Path("tools/docker/Dockerfile")

    # Docker's --build-args is not supported by AzureML Python SDK v2 as shown
    # in [the issue #33902](https://github.com/Azure/azure-sdk-for-python/issues/33902)
    # so the build args are configured by regex substituion
    text = dockerfile.read_text()
    text = re.sub(r"(ARG\sCOMPUTE=).*", rf'\1"{compute}"', text)
    text = re.sub(r"(ARG\sEXTRAS=).*", rf'\1"{extras}"', text)
    text = re.sub(r"(ARG\sGIT_REF=).*", r'\1""', text)
    text = re.sub(r"(ARG\sJDK_VERSION=).*", rf'\1"{conda_openjdk_version}"', text)
    text = re.sub(r"(ARG\sPYTHON_VERSION=).*", rf'\1"{python_version}"', text)
    dockerfile.write_text(text)

    try:
        client.environments.create_or_update(
            Environment(
                name=environment_name,
                build=BuildContext(
                    # Set path for Docker to access to Recommenders root
                    path=".",
                    dockerfile_path=dockerfile,
                ),
            )
        )
    except ResourceExistsError:
        pass


def run_tests(
    client,
    compute,
    environment_name,
    experiment_name,
    script,
    testgroup,
    testkind,
):
    """
    Pytest on AzureML compute.
    See https://github.com/Azure/azureml-examples/blob/main/sdk/python/jobs/single-step/debug-and-monitor/debug-and-monitor.ipynb
    """
    job = client.jobs.create_or_update(
        command(
            experiment_name=experiment_name,
            compute=compute,
            environment=f"{environment_name}@latest",
            code="./",
            command=(
                f"python {script} "
                f"--expname {experiment_name} "
                f"--testgroup {testgroup} "
                f"--testkind {testkind}"
            ),
        )
    )
    client.jobs.stream(job.name)
    job = client.jobs.get(job.name)
    if job.status != "Completed":
        raise JobException("Job Not Completed!")


def correct_resource_name(resource_name):
    """
    Resource name can only contain alphanumeric characters, dashes, and
    underscores, with a limit of 255 characters.
    """
    name = resource_name.replace(".", "_")
    name = name.replace("/", "_")
    return name
