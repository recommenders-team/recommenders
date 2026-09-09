#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Delete the VM created using Terraform.
#
# The following environment variables must be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * CLOUD_SERVICE_SECRET
#   + It contains the access key secret or private key for the
#     selected cloud service APIs.  For example,
#     - It will be used as ALIBABA_CLOUD_ACCESS_KEY_SECRET for
#       AliCloud.
# * CLOUD_SERVICE_ENVS
#   + It may contain the following keys in JSON:
#     - ALIBABA_CLOUD_ACCESS_KEY_ID (required when using AliCloud)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"

[[ -z ${CLOUD_SERVICE:-} ]] && exit 1

cloud_service="${CLOUD_SERVICE}"
cloud_service="${cloud_service@L}"
config_dir="${script_dir}/${cloud_service}"
config_yml="${config_dir}/config.yml"
tf_config_dir="${config_dir}/tf"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "${utils_sh}"

echo 'Exporting environment variables ...'
secret_key_name="$(yq '.secret_key_name' "${config_yml}")"
export "${secret_key_name}"="${CLOUD_SERVICE_SECRET}"
eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"


#--------------------------------------------------------------------
# Use Terraform to delete the VM
#--------------------------------------------------------------------
echo 'Deleting the VM ...'
terraform -chdir="${tf_config_dir}" destroy -auto-approve
