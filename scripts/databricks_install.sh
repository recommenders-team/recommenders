# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------
# This script installs Recommenders into Azure Databricks

echo "Preparing Recommenders library file (egg)..."
zip -r -q Recommenders.egg . -i *.py -x tests/\* scripts/\*

echo "Uploading to databricks..."
dbfs cp --overwrite Recommenders.egg dbfs:/FileStore/Recommenders.egg

# Cluster id should be passed by the first argument
# Cluster id can be found in the URL at https://<databricks-instance>/?o=<16-digit-number>#/setting/clusters/$CLUSTER_ID/configuration.
CLUSTER_ID=$1

echo "Installing the library onto databricks cluster $CLUSTER_ID..."
databricks libraries install --cluster-id $CLUSTER_ID --egg dbfs:/FileStore/Recommenders.egg

databricks libraries cluster-status --cluster-id $CLUSTER_ID

# Restart cluster to make the library active (need when upgrade the library)
echo "Restarting the cluster... will take few seconds. Please check the result from Databricks workspace"
databricks clusters restart --cluster-id $CLUSTER_ID

rm Recommenders.egg
