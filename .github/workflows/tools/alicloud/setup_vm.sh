#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create an AliCloud VM and prepare the environment for testing.
# 
# Params:
# * VM name
# * Test type
#
# The following environment variables must be set:
# * CLOUD_SERVICE
#   + It should be the name of parent directory.
# * CLOUD_SERVICE_SECRET
#   + It contains the access key secret for AliCloud APIs and is used
#     as ALIBABA_CLOUD_ACCESS_KEY_SECRET in the script.
# * CLOUD_SERVICE_EXTRA_DATA
#   + It contains the following keys in JSON:
#     - ALIBABA_CLOUD_ACCESS_KEY_ID (required)
#     - VM_DOCKER_MIRROR_URL (optional)
#       * Semicolon separated URLs of docker mirrors
#     - VM_HTTP_PROXY (optional)
#     - VM_HTTPS_PROXY (optional)
#     - VM_PIP_INDEX_URL (optional)
#     - VM_PROXY_CERTIFICATE (optional)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
vm_name="${1:-}"
test_type="${2:-}"
[[ -z ${vm_name} || -z ${test_type} ]] && exit 1

config_yml="${script_dir}/config.yml"
script_utils="${script_dir}/../utils.sh"
tf_config_dir="${script_dir}/tf"
tools_dir="${script_dir}/../../tools"


echo 'Importing utility functions ...'
source "${script_utils}"

export ALIBABA_CLOUD_ACCESS_KEY_SECRET="${CLOUD_SERVICE_SECRET}"
eval "$(jq -r 'to_entries | .[] | "export \(.key)=\(.value | @sh)"' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"


#--------------------------------------------------------------------
# Use Terraform to create a VM
#--------------------------------------------------------------------
echo 'Creating a VM ...'
terraform -chdir="${tf_config_dir}" init
terraform -chdir="${tf_config_dir}" apply \
    -auto-approve \
    -var "vm_name=${vm_name}" \
    -target 'alicloud_ecs_key_pair.cache' \
    -target 'alicloud_ecs_key_pair_attachment.cache'
vm_ip="$(terraform -chdir="${tf_config_dir}" output \
    -json public_ips \
    | jq -r '.cache')"
ssh_dest="root@${vm_ip}"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"

unset ALIBABA_CLOUD_ACCESS_KEY_SECRET

wait_for_vm_to_be_available "${ssh_dest}"
ssh_key="${tf_config_dir}/${vm_name}"


#--------------------------------------------------------------------
# Basic setup on the VM.
#--------------------------------------------------------------------
echo 'Uploading tools to the VM ...'
scp -qr -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    -i "${ssh_key}" \
    "${tools_dir}" "${ssh_dest}":

readarray -d '' post_create_scripts < \
    <(yq -0 '.scripts.post_create[].script' "${config_yml}")
readarray -d '' reboot_required < \
    <(yq -0 '.scripts.post_create[].reboot' "${config_yml}")
service_tools_dir="${tools_dir##*/}/${CLOUD_SERVICE@L}"
for index in "${!post_create_scripts[@]}"; do
    script="${service_tools_dir}/${post_create_scripts[${index}]}"

    wait_for_vm_to_be_available "${ssh_dest}"
    echo "Running ${script} on the VM ..."
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i "${ssh_key}" \
        "${ssh_dest}" "\
            export VM_DOCKER_MIRROR_URL='${VM_DOCKER_MIRROR_URL:-}'; \
            bash ./${script}"

    if [[ ${reboot_required[${index}]} == true ]]; then
        echo 'Rebooting for setup to take effect ...'
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -i "${ssh_key}" \
            "${ssh_dest}" "sudo reboot" || true
        wait_for_vm_to_be_available "${ssh_dest}"
    fi
done
