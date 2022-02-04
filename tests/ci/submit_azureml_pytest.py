# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
This python script sets up an environment on AzureML and submits a
script to it to run pytest.  It is usually intended to be used as
part of a DevOps pipeline which runs testing on a github repo but
can also be used from command line.

Many parameters are set to default values and some are expected to be passed
in from either the DevOps pipeline or command line.
If calling from command line, there are some parameters you must pass in for
your job to run.


Args:
    Required:
    --clustername (str): the Azure cluster for this run. It can already exist
                         or it will be created.
    --subid       (str): the Azure subscription id

    Optional but suggested, this info will be stored on Azure as
    text information as part of the experiment:
    --pr          (str): the Github PR number
    --reponame    (str): the Github repository name
    --branch      (str): the branch being run
                    It is also possible to put any text string in these.
Example:
    Usually, this script is run by a DevOps pipeline. It can also be
    run from cmd line.
    >>> python tests/ci/refac.py --clustername 'cluster-d3-v2'
                                 --subid '12345678-9012-3456-abcd-123456789012'
                                 --pr '666'
                                 --reponame 'Recommenders'
                                 --branch 'staging'
"""
import argparse
import logging

from azureml.core.authentication import AzureCliAuthentication
from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.script_run_config import ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.workspace import WorkspaceException

EXTRAS = {
    "examples": [
        "azure.mgmt.cosmosdb>=0.8.0,<1",
        "hyperopt>=0.1.2,<1",
        "ipykernel>=4.6.1,<7",
        "jupyter>=1,<2",
        "locust>=1,<2",
        "papermill>=2.1.2,<3",
        "scrapbook>=0.5.0,<1.0.0",
    ],
    "gpu": [
        "nvidia-ml-py3>=7.352.0",
        # TensorFlow compiled with CUDA 11.2, cudnn 8.1
        "tensorflow~=2.6.1;python_version=='3.6'",
        "tensorflow~=2.7.0;python_version>='3.7'",
        "tf-slim>=1.1.0",
        "torch>=1.8",  # for CUDA 11 support
        "fastai>=1.0.46,<2",
    ],
    "spark": [
        "databricks_cli>=0.8.6,<1",
        "pyarrow>=0.12.1,<7.0.0",
        "pyspark>=2.4.5,<4.0.0",
    ],
    "dev": [
        "black>=18.6b4,<21",
        "pytest>=3.6.4",
        "pytest-cov>=2.12.1",
        "pytest-mock>=3.6.1",  # for access to mock fixtures in pytest
        "pytest-rerunfailures>=10.2",  # to mark flaky tests
        "scikit-surprise",
        "pymanopt@https://github.com/pymanopt/pymanopt/archive/fb36a272cdeecb21992cfd9271eb82baafeb316d.zip",
    ],
    "experimental": [
        # xlearn requires cmake to be pre-installed
        "xlearn==0.40a1",
        # VW C++ binary needs to be installed manually for some code to work
        "vowpalwabbit>=8.9.0,<9",
    ],
    "nni": [
        # nni needs to be upgraded
        "nni==1.5",
    ]
}

def setup_workspace(
    workspace_name, subscription_id, resource_group, cli_auth, location
):
    """
    This sets up an Azure Workspace.
    An existing Azure Workspace is used or a new one is created if needed for
    the pytest run.

    Args:
        workspace_name  (str): Centralized location on Azure to work
                               with all the artifacts used by AzureML
                               service
        subscription_id (str): the Azure subscription id
        resource_group  (str): Azure Resource Groups are logical collections of
                         assets associated with a project. Resource groups
                         make it easy to track or delete all resources
                         associated with a project by tracking or deleting
                         the Resource group.
        cli_auth         Azure authentication
        location        (str): workspace reference

    Returns:
        ws: workspace reference
    """
    logger.debug("setup: workspace_name is {}".format(workspace_name))
    logger.debug("setup: resource_group is {}".format(resource_group))
    logger.debug("setup: subid is {}".format(subscription_id))
    logger.debug("setup: location is {}".format(location))

    try:
        # use existing workspace if there is one
        ws = Workspace.get(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            auth=cli_auth,
        )
    except WorkspaceException:
        # this call might take a minute or two.
        logger.debug("Creating new workspace")
        ws = Workspace.create(
            name=workspace_name,
            subscription_id=subscription_id,
            resource_group=resource_group,
            # create_resource_group=True,
            location=location,
            auth=cli_auth,
        )
    return ws


def setup_persistent_compute_target(workspace, cluster_name, vm_size, max_nodes):
    """
    Set up a persistent compute target on AzureML.
    A persistent compute target runs noticeably faster than a
    regular compute target for subsequent runs.  The benefit
    is that AzureML manages turning the compute on/off as needed for
    each job so the user does not need to do this.

    Args:
        workspace    (str): Centralized location on Azure to work with
                         all the
                                artifacts used by AzureML service
        cluster_name (str): the Azure cluster for this run. It can
                            already exist or it will be created.
        vm_size      (str): Azure VM size, like STANDARD_D3_V2
        max_nodes    (int): Number of VMs, max_nodes=4 will
                            autoscale up to 4 VMs
    Returns:
        cpu_cluster : cluster reference
    """
    # setting vmsize and num nodes creates a persistent AzureML
    # compute resource

    logger.debug("setup: cluster_name {}".format(cluster_name))
    # https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-set-up-training-targets

    try:
        cpu_cluster = ComputeTarget(workspace=workspace, name=cluster_name)
        logger.debug("setup: Found existing cluster, use it.")
    except ComputeTargetException:
        logger.debug("setup: create cluster")
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, max_nodes=max_nodes
        )
        cpu_cluster = ComputeTarget.create(workspace, cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=True)
    return cpu_cluster


def create_run_config(cpu_cluster,
                    docker_proc_type,
                    workspace,
                    add_gpu_dependencies,
                    add_spark_dependencies,
                    reco_wheel_path):
    """
    AzureML requires the run environment to be setup prior to submission.
    This configures a docker persistent compute.  Even though
    it is called Persistent compute, AzureML handles startup/shutdown
    of the compute environment.

    Args:
            cpu_cluster      (str)          : Names the cluster for the test
                                                In the case of unit tests, any of
                                                the following:
                                                - Reco_cpu_test
                                                - Reco_gpu_test
            docker_proc_type (str)          : processor type, cpu or gpu
            workspace                       : workspace reference
            add_gpu_dependencies (bool)     : True if gpu packages should be
                                        added to the conda environment, else False
            add_spark_dependencies (bool)   : True if PySpark packages should be
                                        added to the conda environment, else False
    Return:
          run_amlcompute : AzureML run config
    """

    # runconfig with max_run_duration_seconds did not work, check why:
    # run_amlcompute = RunConfiguration(max_run_duration_seconds=60*30)
    run_amlcompute = RunConfiguration()
    run_amlcompute.target = cpu_cluster
    run_amlcompute.environment.docker.enabled = True
    run_amlcompute.environment.docker.base_image = docker_proc_type

    # Use conda_dependencies.yml to create a conda environment in
    # the Docker image for execution
    # False means the user will provide a conda file for setup
    # True means the user will manually configure the environment
    run_amlcompute.environment.python.user_managed_dependencies = False

    # install local version of recommenders on AML compute using .whl file
    whl_url = run_amlcompute.environment.add_private_pip_wheel(
        workspace=workspace,
        file_path=reco_wheel_path,
        exist_ok=True,
    )
    conda_dep = CondaDependencies()
    conda_dep.add_pip_package(whl_url)

    # TODO: install extra dependencies
    extra_dependencies = list(EXTRAS['examples']) + list(EXTRAS['dev'])
    if add_gpu_dependencies:
        extra_dependencies.extend(list(EXTRAS['gpu']))
    if add_spark_dependencies:
        extra_dependencies.extend(list(EXTRAS['spark']))
    for e in extra_dependencies:
        conda_dep.add_pip_package(e)

    run_amlcompute.environment.python.conda_dependencies = conda_dep
    return run_amlcompute


def create_experiment(workspace, experiment_name):
    """
    AzureML requires an experiment as a container of trials.
    This will either create a new experiment or use an
    existing one.

    Args:
        workspace (str) : name of AzureML workspace
        experiment_name (str) : AzureML experiment name
    Return:
        exp - AzureML experiment
    """

    logger.debug("create: experiment_name {}".format(experiment_name))
    exp = Experiment(workspace=workspace, name=experiment_name)
    return exp


def submit_experiment_to_azureml(
    test, test_folder, test_markers, junitxml, run_config, experiment
):

    """
    Submitting the experiment to AzureML actually runs the script.

    Args:
        test         (str) - pytest script, folder/test
                             such as ./tests/ci/run_pytest.py
        test_folder  (str) - folder where tests to run are stored,
                             like ./tests/unit
        test_markers (str) - test markers used by pytest
                             "not notebooks and not spark and not gpu"
        junitxml     (str) - file of output summary of tests run
                             note "--junitxml" is required as part
                             of the string
                             Example: "--junitxml reports/test-unit.xml"
        run_config - environment configuration
        experiment - instance of an Experiment, a collection of
                     trials where each trial is a run.
    Return:
          run : AzureML run or trial
    """

    logger.debug("submit: testfolder {}".format(test_folder))
    logger.debug("junitxml: {}".format(junitxml))
    project_folder = "."

    script_run_config = ScriptRunConfig(
        source_directory=project_folder,
        script=test,
        run_config=run_config,
        arguments=[
            "--testfolder",
            test_folder,
            "--testmarkers",
            test_markers,
            "--xmlname",
            junitxml,
        ],
    )
    run = experiment.submit(script_run_config)
    # waits only for configuration to complete
    run.wait_for_completion(show_output=True, wait_post_processing=True)

    # test logs can also be found on azure
    # go to azure portal to see log in azure ws and look for experiment name
    # and look for individual run
    logger.debug("files {}".format(run.get_file_names))

    return run


def create_arg_parser():
    """
    Many of the argument defaults are used as arg_parser makes it easy to
    use defaults. The user has many options they can select.
    """

    parser = argparse.ArgumentParser(description="Process some inputs")
    # script to run pytest
    parser.add_argument(
        "--test",
        action="store",
        default="./tests/ci/run_pytest.py",
        help="location of script to run pytest",
    )
    # test folder
    parser.add_argument(
        "--testfolder",
        action="store",
        default="./tests/unit",
        help="folder where tests are stored",
    )
    # pytest test markers
    parser.add_argument(
        "--testmarkers",
        action="store",
        default="not notebooks and not spark and not gpu",
        help="pytest markers indicate tests to run",
    )
    # test summary file
    parser.add_argument(
        "--junitxml",
        action="store",
        default="reports/test-unit.xml",
        help="file for returned test results",
    )
    # max num nodes in Azure cluster
    parser.add_argument(
        "--maxnodes",
        action="store",
        default=4,
        help="specify the maximum number of nodes for the run",
    )
    # Azure resource group
    parser.add_argument(
        "--rg", action="store", default="recommender", help="Azure Resource Group"
    )
    # AzureML workspace Name
    parser.add_argument(
        "--wsname", action="store", default="RecoWS", help="AzureML workspace name"
    )
    # AzureML clustername
    parser.add_argument(
        "--clustername",
        action="store",
        default="amlcompute",
        help="Set name of Azure cluster",
    )
    # Azure VM size
    parser.add_argument(
        "--vmsize",
        action="store",
        default="STANDARD_D3_V2",
        help="Set the size of the VM either STANDARD_D3_V2",
    )
    # cpu or gpu
    parser.add_argument(
        "--dockerproc",
        action="store",
        default="cpu",
        help="Base image used in docker container",
    )
    # Azure subscription id, when used in a pipeline, it is stored in keyvault
    parser.add_argument(
        "--subid", action="store", default="123456", help="Azure Subscription ID"
    )
    # reco wheel is created in the azure devops pipeline.
    # Not recommended to change this.
    parser.add_argument(
        "--wheelfile",
        action="store",
        default="./dist/recommenders-1.0.0-py3-none-any.whl",
        help="recommenders whl file path",
    )
    # AzureML experiment name
    parser.add_argument(
        "--expname",
        action="store",
        default="persistentAML",
        help="experiment name on Azure",
    )
    # Azure datacenter location
    parser.add_argument("--location", default="EastUS", help="Azure location")
    # github repo, stored in AzureML experiment for info purposes
    parser.add_argument(
        "--reponame",
        action="store",
        default="--reponame MyGithubRepo",
        help="GitHub repo being tested",
    )
    # github branch, stored in AzureML experiment for info purposes
    parser.add_argument(
        "--branch",
        action="store",
        default="--branch MyGithubBranch",
        help=" Identify the branch test test is run on",
    )
    # github pull request, stored in AzureML experiment for info purposes
    parser.add_argument(
        "--pr",
        action="store",
        default="--pr PRTestRun",
        help="If a pr triggered the test, list it here",
    )
    parser.add_argument(
        "--add_gpu_dependencies", action="store_false", help="include packages for GPU support"
    )
    parser.add_argument(
        "--add_spark_dependencies", action="store_false", help="include packages for PySpark support"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    logger = logging.getLogger("submit_azureml_pytest.py")
    # logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    args = create_arg_parser()

    if args.dockerproc == "cpu":
        from azureml.core.runconfig import DEFAULT_CPU_IMAGE

        docker_proc_type = DEFAULT_CPU_IMAGE
    else:
        from azureml.core.runconfig import DEFAULT_GPU_IMAGE

        docker_proc_type = DEFAULT_GPU_IMAGE

    cli_auth = AzureCliAuthentication()

    workspace = setup_workspace(
        workspace_name=args.wsname,
        subscription_id=args.subid,
        resource_group=args.rg,
        cli_auth=cli_auth,
        location=args.location,
    )

    cpu_cluster = setup_persistent_compute_target(
        workspace=workspace,
        cluster_name=args.clustername,
        vm_size=args.vmsize,
        max_nodes=args.maxnodes,
    )

    run_config = create_run_config(
        cpu_cluster=cpu_cluster,
        docker_proc_type=docker_proc_type,
        workspace=workspace,
        add_gpu_dependencies=args.add_gpu_dependencies,
        add_spark_dependencies=args.add_spark_dependencies,
        reco_wheel_path=args.wheelfile
    )

    logger.info("exp: In Azure, look for experiment named {}".format(args.expname))

    # create new or use existing experiment
    experiment = Experiment(workspace=workspace, name=args.expname)
    run = submit_experiment_to_azureml(
        test=args.test,
        test_folder=args.testfolder,
        test_markers=args.testmarkers,
        junitxml=args.junitxml,
        run_config=run_config,
        experiment=experiment,
    )

    # add helpful information to experiment on Azure
    run.tag("RepoName", args.reponame)
    run.tag("Branch", args.branch)
    run.tag("PR", args.pr)
    # download files from AzureML
    run.download_files(prefix="reports", output_paths="./reports")
    run.complete()
