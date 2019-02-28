#!/usr/bin/env python

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This script installs Recommenders/reco_utils as an egg library onto a Databricks Workspace
# Optionally, also installs a version of mmlspark as a maven library, and prepares the cluster
# for operationalizations

import argparse
import textwrap
import os

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
""

## Variables for operationalization:
COSMOSDB_JAR_FILE_OPTIONS = {
    "3": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.2.0_2.11/1.1.1/azure-cosmosdb-spark_2.2.0_2.11-1.1.1-uber.jar",
    "4": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.2/azure-cosmosdb-spark_2.3.0_2.11-1.2.2-uber.jar",
    "5": "https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.4.0_2.11/1.3.5/azure-cosmosdb-spark_2.4.0_2.11-1.3.5-uber.jar",
}

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

    ## make sure extension is .egg
    if local_eggname[-4:] != ".egg":
        print("appending .egg to end of name.")
        logal_eggname = local_eggname + ".egg"

    ## overwrite egg if it previously existed
    if os.path.exists(local_eggname) and overwrite:
        os.unlink(local_eggname)
    os.rename(myzipfile, local_eggname)
    return local_eggname


def prepare_for_operationalization(
    cluster_id,
    api_client,
    status=None,
    dbfs_path="dbfs:/FileStore/jars",
    overwrite=False,
):
    """
    Installs appropriate versions of several libraries to support operationalization.
    """
    print("Preparing for operationliazation...")
    if status is None:
        ## get status if None
        status = ClusterApi(api_client).get_cluster(cluster_id)

    spark_version = status["spark_version"][0]
    cosmosdb_jar_url = COSMOSDB_JAR_FILE_OPTIONS[spark_version]

    ## download the cosmosdb jar
    local_jarname = os.path.basename(cosmosdb_jar_url)
    ## only download if you need it:
    if overwrite or not os.path.exists(local_jarname):
        print("Downloading {}...".format(cosmosdb_jar_url))
        local_jarname, _ = urlretrieve(cosmosdb_jar_url, local_jarname)
    else:
        print("File {} already downloaded.".format(local_jarname))

    ## upload to dbfs:
    uploadpath = "/".join([dbfs_path, local_jarname])
    print("Uploading CosmosDB driver to databricks at {}".format(uploadpath))
    DbfsApi(api_client).cp(
        recursive=False, src=local_jarname, dst=uploadpath, overwrite=overwrite
    )

    ## setup the list of libraries to install:
    ## jar library setup
    libs2install = [{"jar": uploadpath}]
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
    parser.add_argument("cluster_id", help="cluster id for the cluster to install on.")
    args = parser.parse_args()

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

    ## upload the egg:
    uploadpath = "/".join([args.dbfs_path, os.path.basename(myegg)])
    print(
        "Uploading {} to databricks at {}".format(os.path.basename(myegg), uploadpath)
    )
    DbfsApi(my_api_client).cp(
        recursive=False, src=myegg, dst=uploadpath, overwrite=args.overwrite
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

    ## install the library:
    print(
        "Installing the reco_utils module onto databricks cluster {}".format(
            args.cluster_id
        )
    )
    libs2install = [{"egg": uploadpath}]
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
            status=status,
            dbfs_path=args.dbfs_path,
            overwrite=args.overwrite,
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


sys.exit()

#######################################################
## STOP HERE
#######################################################

# ## setup the config
# my_cluster_config = {
#   "cluster_name": cluster_name,
#   "node_type_id": node_type_id,
#   "autoscale" : {
#     "min_workers": min_workers,
#     "max_workers": max_workers
#   },
#   "autotermination_minutes": autotermination_minutes,
#   "spark_version": spark_version,
#   "spark_env_vars": {
#     "PYSPARK_PYTHON": "/databricks/python3/bin/python3"
#   }
# }


# # ## Search the list for cluster_name
# #
# # Only create the cluster if a cluster of the same name doesn't already exist...
# #
# # Databricks allows multiple clusters with the same name, so this goes through and checks to see if the same name already exists. If so, this notebook will
# # install to that cluster.


# cluster_ids = [c['cluster_id'] for c in cluster_list if c['cluster_name'] == cluster_name]

# if len(cluster_ids) == 0:
#     print("""
#     no clusters with cluster_name ("""+cluster_name+""") found.
#     Trying to create it...
#     """)
#     ## Post the request...
#     response = requests.post(
#         BASE_URL + "clusters/create",
#         headers = my_header,
#         json=my_cluster_config
#     )
#     cluster_id = response.json()['cluster_id']
# else:
#     print("""
#     Cluster named """+cluster_name+""" found!
#     Using that one.
#     Note: It may not have the same configuration as defined in this notebook,
#           so you may want to use a different name.
#     """)
#     if len(cluster_ids) > 1:
#         print("""Warning: Multiple clusters with the same name found. Using the first identified.""")
#     cluster_id = cluster_ids[0]
#     print(cluster_id)

