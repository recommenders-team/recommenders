#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create an AliCloud VM and prepare the environment for testing.
#
# The script must set the environment variable SSH_DEST into
# $GITHUB_ENV for subsequent steps.
#
# Params:
# * Test type
# * VM name
#
# The following environment variables must be set:
# * CLOUD_SERVICE_SECRET
#   + It contains the access key secret for AliCloud APIs and is used
#     as ALIBABA_CLOUD_ACCESS_KEY_SECRET in the script.
# * CLOUD_SERVICE_EXTRA_DATA
#   + It contains the following keys in JSON:
#     - ALIBABA_CLOUD_ACCESS_KEY_ID (required)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
test_type="${1:-}"
vm_name="${2:-}"
[[ -z ${vm_name} || -z ${test_type} ]] && exit 1

tf_config_dir="${script_dir}/tf"

echo 'Importing utility functions ...'
source "${script_dir}/../utils.sh"

echo 'Exporting environment variables ...'
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="${CLOUD_SERVICE_SECRET}"
eval "$(jq -r 'to_entries | .[] | "export \(.key)=\(.value | @sh)"' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"


#--------------------------------------------------------------------
# Use Terraform to create a VM
#--------------------------------------------------------------------
echo 'Creating a VM ...'
ssh_key="${tf_config_dir}/${vm_name}"
terraform -chdir="${tf_config_dir}" init
terraform -chdir="${tf_config_dir}" apply \
    -auto-approve \
    -var "vm_name=${vm_name}" \
    -target 'alicloud_ecs_key_pair.cache' \
    -target 'alicloud_ecs_key_pair_attachment.cache'

unset ALIBABA_CLOUD_ACCESS_KEY_SECRET

echo 'Exporting VM info for subsequent steps ...'
vm_ip="$(terraform -chdir="${tf_config_dir}" output \
    -json public_ips \
    | jq -r '.cache')"
ssh_dest="root@${vm_ip}"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"

wait_for_vm_to_be_available "${ssh_dest}"
setup_ssh_key "${ssh_dest}" "${ssh_key}"
