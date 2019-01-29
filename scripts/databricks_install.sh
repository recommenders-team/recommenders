# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------
# This script installs Recommenders into Databricks

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

CLUSTER_EXIST=false
while IFS=' ' read -ra ARR; do
    if [ ${ARR[0]} = $CLUSTER_ID ]; then
        CLUSTER_EXIST=true

        STATUS=${ARR[2]}
        STATUS=${STATUS//[^a-zA-Z]/}
        if [ $STATUS = RUNNING ]; then
            echo
            echo "Preparing Recommenders library file (egg)..."
            zip -r -q Recommenders.egg . -i *.py -x tests/\* scripts/\*

            echo
            echo "Uploading to databricks..."
            dbfs cp --overwrite Recommenders.egg dbfs:/FileStore/jars/Recommenders.egg

            echo
            echo "Installing the library onto databricks cluster $CLUSTER_ID..."
            databricks libraries install --cluster-id $CLUSTER_ID --egg dbfs:/FileStore/jars/Recommenders.egg

            echo
            echo "Done! Installation status checking..."
            databricks libraries cluster-status --cluster-id $CLUSTER_ID

            echo
            echo "Restarting the cluster to activate the library..."
            databricks clusters restart --cluster-id $CLUSTER_ID

            echo "This will take few seconds. Please check the result from Databricks workspace."
            echo "Alternatively, run 'databricks clusters list' to check the restart status and"
            echo "run 'databricks libraries cluster-status --cluster-id $CLUSTER_ID' to check the installation status."

            rm Recommenders.egg
            exit 0
        else
            echo "Cluster $CLUSTER_ID found, but it is not running. Status=${STATUS}"
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

