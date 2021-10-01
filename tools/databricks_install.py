#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script installs Recommenders/recommenders from PyPI onto a Databricks Workspace
# Optionally, also installs a version of mmlspark as a maven library, and prepares the cluster
# for operationalization

import argparse
import textwrap
import os
from pathlib import Path

import pkg_resources
import sys
import time
from urllib.request import urlretrieve
from requests.exceptions import HTTPError

# requires databricks-cli to be installed and authentication to be configured
from databricks_cli.configure.provider import ProfileConfigProvider
from databricks_cli.configure.config import _get_api_client
from databricks_cli.clusters.api import ClusterApi
from databricks_cli.dbfs.api import DbfsApi
from databricks_cli.libraries.api import LibrariesApi
from databricks_cli.dbfs.dbfs_path import DbfsPath

from recommenders.utils.spark_utils import MMLSPARK_PACKAGE, MMLSPARK_REPO

CLUSTER_NOT_FOUND_MSG = """
    Cannot find the target cluster {}. Please check if you entered the valid id. 
    Cluster id can be found by running 'databricks clusters list', which returns a table formatted as:

    <CLUSTER_ID>\t<CLUSTER_NAME>\t<STATUS>
    """

CLUSTER_NOT_RUNNING_MSG = """
    Cluster {0} found, but it is not running. Status={1}
    You can start the cluster with 'databricks clusters start --cluster-id {0}'.
    Then, check the cluster status by using 'databricks clusters list' and
    re-try installation once the status becomes 'RUNNING'.
    """

# Variables for operationalization:
COSMOSDB_JAR_FILE_OPTIONS = {
    "3": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.2.0_2.11/1.1.1/azure-cosmosdb-spark_2.2.0_2.11-1.1.1-uber.jar",
    "4": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.2/azure-cosmosdb-spark_2.3.0_2.11-1.2.2-uber.jar",
    "5": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.4.0_2.11/1.3.5/azure-cosmosdb-spark_2.4.0_2.11-1.3.5-uber.jar",
    "6": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.4.0_2.11/3.7.0/azure-cosmosdb-spark_2.4.0_2.11-3.7.0-uber.jar",
    "7": "https://search.maven.org/remotecontent?filepath=com/azure/cosmos/spark/azure-cosmos-spark_3-1_2-12/4.3.1/azure-cosmos-spark_3-1_2-12-4.3.1.jar",
    "8": "https://search.maven.org/remotecontent?filepath=com/azure/cosmos/spark/azure-cosmos-spark_3-1_2-12/4.3.1/azure-cosmos-spark_3-1_2-12-4.3.1.jar",
    "9": "https://search.maven.org/remotecontent?filepath=com/azure/cosmos/spark/azure-cosmos-spark_3-1_2-12/4.3.1/azure-cosmos-spark_3-1_2-12-4.3.1.jar"
}

MMLSPARK_INFO = {
    "maven": {
        "coordinates": MMLSPARK_PACKAGE,
        "repo": MMLSPARK_REPO,
    }
}

DEFAULT_CLUSTER_CONFIG = {
    "cluster_name": "DB_CLUSTER",
    "node_type_id": "Standard_D3_v2",
    "autoscale": {"min_workers": 2, "max_workers": 8},
    "autotermination_minutes": 120,
    "spark_version": "5.2.x-scala2.11",
}

PENDING_SLEEP_INTERVAL = 60  # seconds
PENDING_SLEEP_ATTEMPTS = int(
    5 * 60 / PENDING_SLEEP_INTERVAL
)  # wait a maximum of 5 minutes...

## dependencies from PyPI
PYPI_PREREQS = [
    "pip==21.2.4",
    "setuptools==54.0.0",
    "numpy==1.18.0"
]

PYPI_EXTRA_DEPS = [
    "azure-cli-core==2.0.75",
    "azure-mgmt-cosmosdb==0.8.0",
    "azureml-sdk[databricks]",
    "azure-storage-blob<=2.1.0",
]

PYPI_O16N_LIBS = [
    "pydocumentdb>=2.3.3",
]

## Additional dependencies met below.

def dbfs_file_exists(api_client, dbfs_path):
    """Checks to determine whether a file exists.

    Args:
        api_client (ApiClient object): Object used for authenticating to the workspace
        dbfs_path (str): Path to check

    Returns:
        bool: True if file exists on dbfs, False otherwise.
    """
    try:
        DbfsApi(api_client).list_files(dbfs_path=DbfsPath(dbfs_path))
        file_exists = True
    except:
        file_exists = False
    return file_exists


def get_installed_libraries(api_client, cluster_id):
    """Returns the installed PyPI packages and the ones that failed.

    Args:
        api_client (ApiClient object): object used for authenticating to the workspace
        cluster_id (str): id of the cluster
    
    Returns:
        Dict[str, str]: dictionary of {package: status} 
    """
    cluster_status = LibrariesApi(api_client).cluster_status(cluster_id)
    libraries = {lib["library"]["pypi"]["package"]: lib["status"] for lib in cluster_status["library_statuses"] if "pypi" in lib["library"]}
    return {pkg_resources.Requirement.parse(package).name: libraries[package] for package in libraries}


def prepare_for_operationalization(
    cluster_id, api_client, dbfs_path, overwrite, spark_version
):
    """
    Installs appropriate versions of several libraries to support operationalization.

    Args:
        cluster_id (str): cluster_id representing the cluster to prepare for operationalization
        api_client (ApiClient): the ApiClient object used to authenticate to the workspace
        dbfs_path (str): the path on dbfs to upload libraries to
        overwrite (bool): whether to overwrite existing files on dbfs with new files of the same name
        spark_version (str): str version indicating which version of spark is installed on the databricks cluster

    Returns:
        A dictionary of libraries installed
    """
    print("Preparing for operationlization...")

    cosmosdb_jar_url = COSMOSDB_JAR_FILE_OPTIONS[spark_version]

    # download the cosmosdb jar
    local_jarname = os.path.basename(cosmosdb_jar_url)
    # only download if you need it:
    if overwrite or not os.path.exists(local_jarname):
        print("Downloading {}...".format(cosmosdb_jar_url))
        local_jarname, _ = urlretrieve(cosmosdb_jar_url, local_jarname)
    else:
        print("File {} already downloaded.".format(local_jarname))

    # upload jar to dbfs:
    upload_path = Path(dbfs_path, local_jarname).as_posix()
    print("Uploading CosmosDB driver to databricks at {}".format(upload_path))
    if dbfs_file_exists(api_client, upload_path) and overwrite:
        print("Overwriting file at {}".format(upload_path))
    DbfsApi(api_client).cp(
        recursive=False, src=local_jarname, dst=upload_path, overwrite=overwrite
    )

    # setup the list of libraries to install:
    # jar library setup
    libs2install = [{"jar": upload_path}]
    # setup libraries to install:
    libs2install.extend([{"pypi": {"package": i}} for i in PYPI_O16N_LIBS])
    print("Installing jar and pypi libraries required for operationalization...")
    LibrariesApi(api_client).install_libraries(cluster_id, libs2install)
    return libs2install


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
      This script installs the recommenders package from PyPI onto a databricks cluster. 
      Optionally, this script may also install the mmlspark library, and it may also install additional libraries useful 
      for operationalization. This script requires that you have installed databricks-cli in the python environment in 
      which you are running this script, and that have you have already configured it with a profile.
      """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--profile",
        help="The CLI profile to use for connecting to the databricks workspace",
        default="DEFAULT",
    )
    parser.add_argument(
        "--path-to-recommenders",
        help="The path to the root of the recommenders repository. Default assumes that the script is run in the root of the repository",
        default=".",
    )
    parser.add_argument(
        "--dbfs-path",
        help="The directory on dbfs that want to place files in",
        default="dbfs:/FileStore/jars",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite existing files."
    )
    parser.add_argument(
        "--prepare-o16n",
        action="store_true",
        help="Whether to install additional libraries for operationalization.",
    )
    parser.add_argument(
        "--mmlspark", action="store_true", help="Whether to install mmlspark."
    )
    parser.add_argument(
        "--create-cluster",
        action="store_true",
        help="Whether to create the cluster. This will create a cluster with default parameters.",
    )
    parser.add_argument(
        "cluster_id",
        help="cluster id for the cluster to install data on. If used in conjunction with --create-cluster, this is the name of the cluster created",
    )
    args = parser.parse_args()

    # make sure path_to_recommenders is on sys.path to allow for import
    sys.path.append(args.path_to_recommenders)

    ############################
    # Interact with Databricks:
    ############################

    # first make sure you are using the correct profile and connecting to the intended workspace
    my_api_client = _get_api_client(ProfileConfigProvider(args.profile).get_config())

    # Create a cluster if flagged
    if args.create_cluster:
        # treat args.cluster_id as the name, because if you create a cluster, you do not know its id yet.
        DEFAULT_CLUSTER_CONFIG["cluster_name"] = args.cluster_id
        cluster_info = ClusterApi(my_api_client).create_cluster(DEFAULT_CLUSTER_CONFIG)
        args.cluster_id = cluster_info["cluster_id"]
        print(
            "Creating a new cluster with name {}. New cluster_id={}".format(
                DEFAULT_CLUSTER_CONFIG["cluster_name"], args.cluster_id
            )
        )

    # steps below require the cluster to be running. Check status
    try:
        status = ClusterApi(my_api_client).get_cluster(args.cluster_id)
    except HTTPError as e:
        print(e)
        print(textwrap.dedent(CLUSTER_NOT_FOUND_MSG.format(args.cluster_id)))
        raise

    if status["state"] == "TERMINATED":
        print(
            textwrap.dedent(
                CLUSTER_NOT_RUNNING_MSG.format(args.cluster_id, status["state"])
            )
        )
        sys.exit()

    attempt = 0
    while status["state"] == "PENDING" and attempt < PENDING_SLEEP_ATTEMPTS:
        print(
            "Current status=={}... Waiting {}s before trying again (attempt {}/{}).".format(
                status["state"],
                PENDING_SLEEP_INTERVAL,
                attempt + 1,
                PENDING_SLEEP_ATTEMPTS,
            )
        )
        time.sleep(PENDING_SLEEP_INTERVAL)
        status = ClusterApi(my_api_client).get_cluster(args.cluster_id)
        attempt += 1

    # if it is still PENDING, exit.
    if status["state"] == "PENDING":
        print(
            textwrap.dedent(
                CLUSTER_NOT_RUNNING_MSG.format(args.cluster_id, status["state"])
            )
        )
        sys.exit()


    # install prerequisites
    print(
        "Installing required Python libraries onto databricks cluster {}".format(
            args.cluster_id
        )
    )
    libs2install = [{"pypi": {"package": i}} for i in PYPI_PREREQS]
    LibrariesApi(my_api_client).install_libraries(args.cluster_id, libs2install)

    # install the library and its dependencies
    print(
        "Installing the recommenders package onto databricks cluster {}".format(
            args.cluster_id
        )
    )
    LibrariesApi(my_api_client).install_libraries(args.cluster_id, [{"pypi": {"package": "recommenders"}}])

    # pip cannot handle everything together, so wait until recommenders package is installed
    installed_libraries = get_installed_libraries(my_api_client, args.cluster_id)
    while "recommenders" not in installed_libraries:
        time.sleep(PENDING_SLEEP_INTERVAL)
        installed_libraries = get_installed_libraries(my_api_client, args.cluster_id)
    while installed_libraries["recommenders"] not in ["INSTALLED", "FAILED"]:
        time.sleep(PENDING_SLEEP_INTERVAL)
        installed_libraries = get_installed_libraries(my_api_client, args.cluster_id)
    if installed_libraries["recommenders"] == "FAILED":
        raise Exception("recommenders package failed to install")    

    # additional PyPI dependencies:
    libs2install = [{"pypi": {"package": i}} for i in PYPI_EXTRA_DEPS]

    # add mmlspark if selected.
    if args.mmlspark:
        print("Installing MMLSPARK package...")
        libs2install.extend([MMLSPARK_INFO])
    print(
        "Installing {} onto databricks cluster {}".format(
            libs2install,
            args.cluster_id
        )
    )
    LibrariesApi(my_api_client).install_libraries(args.cluster_id, libs2install)

    # prepare for operationalization if desired:
    if args.prepare_o16n:
        prepare_for_operationalization(
            cluster_id=args.cluster_id,
            api_client=my_api_client,
            dbfs_path=args.dbfs_path,
            overwrite=args.overwrite,
            spark_version=status["spark_version"][0],
        )

    # restart the cluster for new installation(s) to take effect.
    print("Restarting databricks cluster {}".format(args.cluster_id))
    ClusterApi(my_api_client).restart_cluster(args.cluster_id)

    # wrap up and send out a final message:
    print(
        """
        Requests submitted. You can check on status of your cluster with: 

        databricks --profile """
        + args.profile
        + """ clusters list
        """
    )
