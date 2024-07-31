# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
This module includes utilities for tests on AzureML via AML Python SDK v2.
See
* https://learn.microsoft.com/en-us/azure/machine-learning/concept-v2?view=azureml-api-2
* https://learn.microsoft.com/en-us/azure/machine-learning/reference-migrate-sdk-v1-mlflow-tracking?view=azureml-api-2&tabs=aml%2Ccli%2Cmlflow
"""
import pathlib
import tempfile

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
    conda_pkg_jdk,
    python_version,
    commit_sha,
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
        conda_pkg_jdk (str): "openjdk=8" by default
        python_version (str): python version, such as "3.9"
        commit_sha (str): the commit that triggers the workflow
    """
    conda_env_name = "reco"
    conda_env_yml = "environment.yml"
    condafile = fr"""
name: {conda_env_name}
channels:
  - conda-forge
dependencies:
  - python={python_version}
  - {conda_pkg_jdk}
  - pip
  - pip:
    - pymanopt@https://github.com/pymanopt/pymanopt/archive/fb36a272cdeecb21992cfd9271eb82baafeb316d.zip
    - recommenders[dev{",gpu" if use_gpu else ""}{",spark" if use_spark else ""}]@git+https://github.com/recommenders-team/recommenders.git@{commit_sha}
"""
    # See https://github.com/Azure/AzureML-Containers/blob/master/base/cpu/openmpi4.1.0-ubuntu22.04
    image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04"
    # See https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04
    dockerfile = fr"""# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/cuda:12.5.1-devel-ubuntu22.04
SHELL ["/bin/bash", "-c"]
USER root:root
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y wget git-all && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Conda
ENV CONDA_PREFIX /opt/miniconda
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_24.5.0-0-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p ${{CONDA_PREFIX}} && \
    ${{CONDA_PREFIX}}/bin/conda update --all -c conda-forge -y && \
    ${{CONDA_PREFIX}}/bin/conda clean -ay && \
    rm -rf ${{CONDA_PREFIX}}/pkgs && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Create Conda environment
COPY {conda_env_yml} /tmp/{conda_env_yml}
RUN ${{CONDA_PREFIX}}/bin/conda env create -f /tmp/{conda_env_yml}

# Activate Conda environment
ENV CONDA_DEFAULT_ENV {conda_env_name}
ENV CONDA_PREFIX ${{CONDA_PREFIX}}/envs/${{CONDA_DEFAULT_ENV}}
ENV PATH="${{CONDA_PREFIX}}/bin:${{PATH}}"  LD_LIBRARY_PATH="${{CONDA_PREFIX}}/lib:$LD_LIBRARY_PATH"
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = pathlib.Path(tmpdir)
        dockerfile_path = tmpdir / "Dockerfile"
        condafile_path = tmpdir / conda_env_yml
        build = BuildContext(path=tmpdir, dockerfile_path=dockerfile_path.name)

        with open(dockerfile_path, "w") as file:
            file.write(dockerfile)
        with open(condafile_path, "w") as file:
            file.write(condafile)

        try:
            client.environments.create_or_update(
                Environment(
                    name=environment_name,
                    image=None if use_gpu else image,
                    build=build if use_gpu else None,
                    conda_file=None if use_gpu else condafile_path,
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
