# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

"""
This python script sets up an environment on AzureML and submits a
script to it to run pytest.  It is usually intended to be used as
part of a DevOps pipeline which runs testing on a GitHub repo but
can also be used from command line.

Many parameters are set to default values and some are expected to be passed
in from either the DevOps pipeline or command line.
If calling from command line, there are some parameters you must pass in for
your job to run.


Args:
    See parse_args() below for more details.

Example:
    Usually, this script is run by a DevOps pipeline. It can also be
    run from cmd line.
    >>> python tests/ci/submit_groupwise_azureml_pytest.py \
            --subid '12345678-9012-3456-abcd-123456789012' ...
"""
import argparse
import logging

from aml_utils import (
    correct_resource_name,
    create_or_start_compute,
    get_client,
    get_or_create_environment,
    run_tests,
)


def parse_args():
    """
    Many of the argument defaults are used as arg_parser makes it easy to
    use defaults. The user has many options they can select.
    """

    parser = argparse.ArgumentParser(description="Process some inputs")

    parser.add_argument(
        "--sha",
        action="store",
        help="the commit triggering the workflow",
    )
    parser.add_argument(
        "--script",
        action="store",
        default="tests/ci/azureml_tests/run_groupwise_pytest.py",
        help="Path of script to run pytest",
    )
    parser.add_argument(
        "--maxnodes",
        action="store",
        default=4,
        help="Maximum number of nodes for the run",
    )
    parser.add_argument(
        "--testgroup",
        action="store",
        default="group_criteo",
        help="Test Group",
    )
    parser.add_argument(
        "--rg",
        action="store",
        default="recommender",
        help="Azure Resource Group",
    )
    parser.add_argument(
        "--ws",
        action="store",
        default="RecoWS",
        help="AzureML workspace name",
    )
    parser.add_argument(
        "--cluster",
        action="store",
        default="azuremlcompute",
        help="AzureML cluster name",
    )
    parser.add_argument(
        "--vmsize",
        action="store",
        default="STANDARD_D3_V2",
        help="VM size",
    )
    parser.add_argument(
        "--subid",
        action="store",
        default="123456",
        help="Azure Subscription ID",
    )
    parser.add_argument(
        "--expname",
        action="store",
        default="persistentAzureML",
        help="Experiment name on AzureML",
    )
    parser.add_argument(
        "--envname",
        action="store",
        default="recommenders",
        help="Environment name on AzureML",
    )
    parser.add_argument(
        "--conda-openjdk-version",
        action="store",
        default="21",
        help="Conda OpenJDK package version",
    )
    parser.add_argument(
        "--python-version",
        action="store",
        default="3.11",
        help="Python version",
    )
    parser.add_argument(
        "--testkind",
        action="store",
        default="unit",
        help="Test kind - nightly or unit",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger("submit_groupwise_azureml_pytest.py")
    args = parse_args()

    logger.info("Setting up workspace %s", args.ws)
    client = get_client(
        subscription_id=args.subid,
        resource_group=args.rg,
        workspace_name=args.ws,
    )

    logger.info("Setting up compute %s", args.cluster)
    create_or_start_compute(
        client=client, name=args.cluster, size=args.vmsize, max_instances=args.maxnodes
    )

    # TODO: Unlike Azure DevOps pipelines, GitHub Actions only has simple
    #       string functions like startsWith() and contains().  And AzureML
    #       only accepts simple names that do not contain '.' and '/'.
    #       correct_resource_name() is used to replace '.' and '/' with '_'
    #       which makes names in the workflow and on AzureML inconsistent.
    #       For example, a name
    #       * in the workflow
    #           recommenders-unit-group_cpu_001-python3.8-c8adeafabc011b549f875dc145313ffbe3fc53a8
    #       * on AzureML
    #           recommenders-unit-group_cpu_001-python3_8-c8adeafabc011b549f875dc145313ffbe3fc53a8
    environment_name = correct_resource_name(args.envname)
    logger.info("Setting up environment %s", environment_name)
    get_or_create_environment(
        client=client,
        environment_name=environment_name,
        use_gpu="gpu" in args.testgroup,
        use_spark="spark" in args.testgroup,
        conda_openjdk_version=args.conda_openjdk_version,
        python_version=args.python_version,
    )

    experiment_name = correct_resource_name(args.expname)
    logger.info("Running experiment %s", experiment_name)
    run_tests(
        client=client,
        compute=args.cluster,
        environment_name=environment_name,
        experiment_name=experiment_name,
        script=args.script,
        testgroup=args.testgroup,
        testkind=args.testkind,
    )
