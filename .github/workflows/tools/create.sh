#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create an VM using Terraform and prepare the environment for testing.
#
# The script must set the environment variable SSH_DEST into
# $GITHUB_ENV for subsequent steps.
#
# Params:
# * VM name
# * Test type
# * Test group
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
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE_INPUT_VARS
#   + It contains the possible values of the input variables for
#     creating the VM, in the JSON format like the following:
#
#     {
#         "cpu": {
#             "instance_type_family": [ "ecs.e" ],
#             "region": [
#                 "ap-southeast-5",
#                 "ap-northeast-1",
#                 "eu-central-1",
#                 "ap-southeast-1"
#             ]
#         },
#         "gpu": {
#             "instance_type_family": [
#                 "ecs.gn8is",
#                 "ecs.gn7i",
#                 "ecs.gn6i"
#             ],
#             "region": [
#                 "ap-southeast-5",
#                 "ap-northeast-1",
#                 "eu-central-1",
#                 "ap-southeast-1"
#             ]
#         }
#     }
#
#     where "cpu" and "gpu" is used for different compute types if
#     needed, or it can simply reduce to
#
#     {
#         "instance_type_family": [
#             "ecs.gn8is",
#             "ecs.gn7i",
#             "ecs.gn6i"
#         ],
#         "region": [
#             "ap-southeast-5",
#             "ap-northeast-1",
#             "eu-central-1",
#             "ap-southeast-1"
#         ]
#     }
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
vm_name="${1:-}"
test_type="${2:-}"
test_group="${3:-}"
[[ -z ${vm_name} \
  || -z ${test_type} \
  || -z ${test_group} \
  || -z ${CLOUD_SERVICE:-} ]] && exit 1

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
# Use Terraform to create a VM
#--------------------------------------------------------------------
echo 'Creating a VM ...'
terraform -chdir="${tf_config_dir}" init

cloud_service_input_vars="${CLOUD_SERVICE_INPUT_VARS:-}"
if [[ ${test_group} == *cpu* ]]; then
    input_vars="$(jq '.cpu // empty' <<< "${cloud_service_input_vars}")"
else
    input_vars="$(jq '.gpu // empty' <<< "${cloud_service_input_vars}")"
fi

if [[ -z ${input_vars} ]]; then
    input_vars="${cloud_service_input_vars}"
fi

apply_tf_config "${vm_name}" "${tf_config_dir}" "${input_vars}"

unset "${secret_key_name}"

echo 'Exporting VM info for subsequent steps ...'
ssh_dest="$(terraform -chdir="${tf_config_dir}" output -json ssh_dest | jq -r)"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"

wait_for_vm_to_be_available "${ssh_dest}"
ssh_key="${tf_config_dir}/${vm_name}"
setup_ssh_key "${ssh_dest}" "${ssh_key}"
