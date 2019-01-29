# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------
# This script installs appropriate external libraries onto
# a databricks cluster for operationalization.

DATABRICKS_CLI=$(which databricks)
if ! [ -x "$DATABRICKS_CLI" ]; then
    echo "No databricks-cli found!! Please see the SETUP.md file for installation prerequisites."
    exit 1
fi

CLUSTER_ID=$1
if [ -z $CLUSTER_ID ]; then
    echo "Please provide the target cluster id: 'databricks_install.sh <CLUSTER_ID>'."
    echo "Cluster id can be found by running 'databricks clusters list'"
    echo "which returns a list of <CLUSTER_ID> <CLUSTER_NAME> <STATUS>."
    exit 1
fi

## for spark version >=2.3.0
COSMOSDB_CONNECTOR_URL="https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.2/azure-cosmosdb-spark_2.3.0_2.11-1.2.2-uber.jar"
COSMOSDB_CONNECTOR_BASENAME=$(basename $COSMOSDB_CONNECTOR_URL)

CLUSTER_EXIST=false
PYPI_LIBRARIES=( "azure-cli" "azureml-sdk[databricks]" "pydocumentdb" )
while IFS=' ' read -ra ARR; do
    if [ ${ARR[0]} = $CLUSTER_ID ]; then
        CLUSTER_EXIST=true
        if [ ${ARR[2]} = RUNNING ]; then
            ## install each of the pypi libraries
            for lib in "${PYPI_LIBRARIES[@]}"
            do
                echo
                echo "Adding $lib"
                echo
                databricks libraries install --cluster-id $CLUSTER_ID --pypi-package $lib
            done

            ## get spark-cosmosdb connector:
            echo
            echo "downloading cosmosdb connector jar file"
            echo
            curl -O $COSMOSDB_CONNECTOR_URL
            
            ## uplaod the jar to dbfs
            echo
            echo "Uploading to dbfs"
            echo
            dbfs cp --overwrite ${COSMOSDB_CONNECTOR_BASENAME} dbfs:/FileStore/jars/${COSMOSDB_CONNECTOR_BASENAME}

            # isntall from dbfs
            echo
            echo "Adding ${COSMOSDB_CONNECTOR_BASENAME} as library"
            echo
            databricks libraries install --cluster-id $CLUSTER_ID --jar dbfs:/FileStore/jars/${COSMOSDB_CONNECTOR_BASENAME}

            ## Check installation status
            echo
            echo "Done! Installation status checking..."
            databricks libraries cluster-status --cluster-id $CLUSTER_ID

            echo
            echo "Restarting the cluster to activate the library..."
            databricks clusters restart --cluster-id $CLUSTER_ID

            echo "This will take few seconds. Please check the result from Databricks workspace."
            echo "Alternatively, run 'databricks clusters list' to check the restart status and"
            echo "run 'databricks libraries cluster-status --cluster-id $CLUSTER_ID' to check the installation status."

            exit 0
        else
            echo "Cluster $CLUSTER_ID found, but it is not running. Status=${ARR[2]}"
            echo "You can start the cluster with 'databricks clusters start --cluster-id $CLUSTER_ID'."
            echo "Then, check the cluster status by using 'databricks clusters list' and"
            echo "re-try installation once the status turns into RUNNING."
            exit 1
        fi
    fi
done < <(databricks clusters list)

if ! [ $CLUSTER_EXIST = true ]; then
    echo "Cannot find the target cluster $CLUSTER_ID. Please check if you entered the valid id."
    echo "Cluster id can be found by running 'databricks clusters list'"
    echo "which returns a list of <CLUSTER_ID> <CLUSTER_NAME> <STATUS>."
    exit 1
fi

