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
# * VM name
# * Test type
# * Test group
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
vm_name="${1:-}"
test_type="${2:-}"
test_group="${3:-}"
[[ -z ${vm_name} || -z ${test_type} || -z ${test_group} ]] && exit 1

tf_config_dir="${script_dir}/tf"


#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "${script_dir}/../utils.sh"

echo 'Exporting environment variables ...'
export ALIBABA_CLOUD_ACCESS_KEY_SECRET="${CLOUD_SERVICE_SECRET}"
eval "$(generate_var_exports "${CLOUD_SERVICE_EXTRA_DATA:-}")"


#--------------------------------------------------------------------
# Use Terraform to create a VM
#--------------------------------------------------------------------
echo 'Creating a VM ...'
terraform -chdir="${tf_config_dir}" init
if [[ ${test_group} == *cpu* ]]; then
    terraform -chdir="${tf_config_dir}" apply \
        -auto-approve \
        -var "vm_name=${vm_name}" \
        -var "instance_type_family=ecs.e"
else
    terraform -chdir="${tf_config_dir}" apply \
        -auto-approve \
        -var "vm_name=${vm_name}" \
        -var "instance_type_family=ecs.gn8is"
fi

unset ALIBABA_CLOUD_ACCESS_KEY_SECRET

echo 'Exporting VM info for subsequent steps ...'
ip="$(terraform -chdir="${tf_config_dir}" output -json ip | jq -r)"
ssh_dest="root@${ip}"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"

wait_for_vm_to_be_available "${ssh_dest}"
setup_ssh_key "${ssh_dest}" "${tf_config_dir}/${vm_name}"
