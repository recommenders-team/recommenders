#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script installs Recommenders/reco_utils as an egg library onto a Databricks Workspace
# Optionally, also installs a version of mmlspark as a maven library, and prepares the cluster
# for operationalization

import argparse
import textwrap
import os
from pathlib import Path

import shutil
import sys
from urllib.request import urlretrieve

## requires databricks-cli to be installed:
## and requires authentication to be configured
from databricks_cli.configure.provider import ProfileConfigProvider
from databricks_cli.configure.config import _get_api_client
from databricks_cli.clusters.api import ClusterApi
from databricks_cli.dbfs.api import DbfsApi
from databricks_cli.libraries.api import LibrariesApi

from requests.exceptions import HTTPError

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

## Variables for operationalization:
COSMOSDB_JAR_FILE_OPTIONS = {
    "3": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.2.0_2.11/1.1.1/azure-cosmosdb-spark_2.2.0_2.11-1.1.1-uber.jar",
    "4": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.2/azure-cosmosdb-spark_2.3.0_2.11-1.2.2-uber.jar",
    "5": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.4.0_2.11/1.3.5/azure-cosmosdb-spark_2.4.0_2.11-1.3.5-uber.jar",
}

PYPI_RECO_LIB_DEPS = ["tqdm==4.31.1"]

PYPI_O16N_LIBS = [
    "azure-cli==2.0.56",
    "azureml-sdk[databricks]==1.0.8",
    "pydocumentdb==2.3.3",
]

MMLSPARK_INFO = {
    "maven": {
        "coordinates": "com.microsoft.ml.spark:mmlspark_2.11:0.15.dev29",
        "repo": "https://mmlspark.azureedge.net/maven",
    }
}

DEFAULT_CLUSTER_CONFIG = {
    "cluster_name": "DB_CLUSTER",
    "node_type_id": "Standard_D3_v2",
    "autoscale" : {
    "min_workers": 2,
    "max_workers": 8
    },
    "autotermination_minutes": 120,
    "spark_version": "5.2.x-scala2.11",
}

def create_egg(
    path_to_recommenders_repo_root=os.getcwd(),
    local_eggname="Recommenders.egg",
    overwrite=False,
):
    """
    Packages files in the reco_utils directory as a .egg file that can be uploaded to dbfs and installed as a library on a databricks cluster.
    """
    ## create the zip archive:
    myzipfile = shutil.make_archive(
        "reco_utils",
        "zip",
        root_dir=path_to_recommenders_repo_root,
        base_dir="reco_utils",
    )

    ## overwrite egg if it previously existed
    if os.path.exists(local_eggname) and overwrite:
        os.unlink(local_eggname)
    os.rename(myzipfile, local_eggname)
    return local_eggname


def prepare_for_operationalization(
    cluster_id, api_client, dbfs_path, overwrite, spark_version
):
    """
    Installs appropriate versions of several libraries to support operationalization.
    """
    print("Preparing for operationlization...")

    cosmosdb_jar_url = COSMOSDB_JAR_FILE_OPTIONS[spark_version]

    ## download the cosmosdb jar
    local_jarname = os.path.basename(cosmosdb_jar_url)
    ## only download if you need it:
    if overwrite or not os.path.exists(local_jarname):
        print("Downloading {}...".format(cosmosdb_jar_url))
        local_jarname, _ = urlretrieve(cosmosdb_jar_url, local_jarname)
    else:
        print("File {} already downloaded.".format(local_jarname))

    ## upload jar to dbfs:
    upload_path = Path(args.dbfs_path, local_jarname).as_posix()
    print("Uploading CosmosDB driver to databricks at {}".format(upload_path))
    DbfsApi(api_client).cp(
        recursive=False, src=local_jarname, dst=upload_path, overwrite=overwrite
    )

    ## setup the list of libraries to install:
    ## jar library setup
    libs2install = [{"jar": upload_path}]
    ## setup libraries to install:
    libs2install.extend([{"pypi": {"package": i}} for i in PYPI_O16N_LIBS])
    print("Installing jar and pypi libraries required for operationalizaiton...")
    LibrariesApi(api_client).install_libraries(args.cluster_id, libs2install)
    return libs2install


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
      This script packages the reco_utils directory into a .egg file and installs it onto a databricks cluster. 
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
        "--eggname",
        help="Name of the egg you want to generate. Useful if you want to name based on branch or date.",
        default="Recommenders.egg",
    )
    parser.add_argument(
        "--dbfs-path",
        help="The directory on dbfs that want to place files in",
        default="dbfs:/FileStore/jars",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files.",
        default=False,
    )
    parser.add_argument(
        "--prepare-o16n",
        action="store_true",
        help="Whether to install additional libraries for operationalization.",
        default=False,
    )
    parser.add_argument(
        "--mmlspark",
        action="store_true",
        help="Whether to install mmlspark.",
        default=False,
    )
    parser.add_argument(
        "--create-cluster",
        help="The name of the cluster to create. This creates a cluster with default parameters.",
        default=None,
    )
    parser.add_argument("cluster_id", help="cluster id for the cluster to install on.")
    args = parser.parse_args()

    ## Check name of egg for extension.
    if not args.eggname.endswith(".egg"):
        args.eggname += ".egg"

    ###############################
    ## Create the egg:
    ###############################

    print("Preparing Recommenders library file ({})...".format(args.eggname))
    myegg = create_egg(
        args.path_to_recommenders, local_eggname=args.eggname, overwrite=args.overwrite
    )
    print("Created: {}".format(myegg))

    ###############################
    ## Interact with Databricks:
    ###############################

    ## first make sure you are using the correct profile and connecting to the intended workspace
    my_api_client = _get_api_client(ProfileConfigProvider(args.profile).get_config())

    ## Check if file exists to alert user.
    ## upload the egg:
    upload_path = Path(args.dbfs_path, args.eggname).as_posix()
    print("Uploading {} to databricks at {}".format(args.eggname, upload_path))
    DbfsApi(my_api_client).cp(
        recursive=False, src=myegg, dst=upload_path, overwrite=args.overwrite
    )

    ## steps below require the cluster to be running. Check status
    try:
        status = ClusterApi(my_api_client).get_cluster(args.cluster_id)
    except HTTPError as e:
        print(e)
        print(textwrap.dedent(CLUSTER_NOT_FOUND_MSG.format(args.cluster_id)))
        raise

    if status["state"] != "RUNNING":
        print(
            textwrap.dedent(
                CLUSTER_NOT_RUNNING_MSG.format(args.cluster_id, status["state"])
            )
        )
        sys.exit()

    ## install the library and its dependencies
    print(
        "Installing the reco_utils module onto databricks cluster {}".format(
            args.cluster_id
        )
    )
    libs2install = [{"egg": upload_path}]
    ## PYPI dependencies:
    libs2install.extend([{"pypi": {"package": i}} for i in PYPI_RECO_LIB_DEPS])

    ## add mmlspark if selected.
    if args.mmlspark:
        print("Installing MMLSPARK package...")
        libs2install.extend([MMLSPARK_INFO])
    print(libs2install)
    LibrariesApi(my_api_client).install_libraries(args.cluster_id, libs2install)

    ## prepare for operationalization if desired:
    if args.prepare_o16n:
        prepare_for_operationalization(
            cluster_id=args.cluster_id,
            api_client=my_api_client,
            dbfs_path=args.dbfs_path,
            overwrite=args.overwrite,
            spark_version=status["spark_version"][0],
        )

    ## restart the cluster for new installation(s) to take effect.
    print("Restarting databricks cluster {}".format(args.cluster_id))
    ClusterApi(my_api_client).restart_cluster(args.cluster_id)

    ## wrap up and send out a final message:
    print(
        """
        Requests submitted. You can check on status of your cluster with: 

        databricks --profile """
        + args.profile
        + """ clusters list
        """
    )

