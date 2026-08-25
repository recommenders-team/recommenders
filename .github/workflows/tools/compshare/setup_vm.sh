#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create a CompShare VM and prepare the environment for testing
# 
# Params:
# * VM name
# * Test type
#
# The following environment variables must be set:
# * CLOUD_SERVICE
#   + It should be the name of parent directory.
# * CLOUD_SERVICE_SECRET
#   + It contains the private key for CompShare APIs and is used as
#     COMPSHARE_PRIVATE_KEY in the script.
# * CLOUD_SERVICE_EXTRA_DATA
#   + It contains the following keys in JSON:
#     - COMPSHARE_PUBLIC_KEY (required)
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
script_utils="${script_dir}/utils.sh"
tools_dir="${script_dir}/../../tools"

echo 'Importing utility functions ...'
source "${script_utils}"

echo 'Exporting environment variables ...'
export COMPSHARE_PRIVATE_KEY="${CLOUD_SERVICE_SECRET}"
eval "$(jq -r 'to_entries | .[] | "export \(.key)=\(.value | @sh)"' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"


#--------------------------------------------------------------------
# Use CompShare APIs to create a VM.
#--------------------------------------------------------------------
# * All VMs from CompShare have GPUs.
# * Unit tests require less than half hour
# * Nightly tests take more than an hour and more GPU memory.
# * VMs with Spot ChargeType are cheaper but there is a risk of being
#   deleted after 1 hour.
if [[ "${test_type}" == *nightly* ]]; then
    gpu_type='"GPUType": "!2080,P40"'
    gpu_mem='"Memory": {"GPU": 12, "CPU": 32}'
    charge_type='"ChargeType": ["Postpay"]'
    stop_time="\"SchedulerStopTime\": $(date --date='3 hours' '+%s')"
else
    gpu_type='"GPUType": "!P40"'
    gpu_mem='"Memory": {"GPU": 8, "CPU": 32}'
    charge_type='"ChargeType": ["Spot","Postpay"]'
    stop_time="\"SchedulerStopTime\": $(date --date='1 hours' '+%s')"
fi
requirements="{${gpu_type}, ${gpu_mem}, ${charge_type}, ${stop_time}}"

echo 'Generating login password ...'
encoded_password_file="$(mktemp)"
mktemp -u XXXXXXXXXX | tr -d '\n' | base64 \
    | tr -d '\n' > "${encoded_password_file}"

allocate_vm \
    "${vm_name}" \
    "${encoded_password_file}" \
    "$(jq 'del(.SchedulerStopTime)' <<< "${requirements}")"

echo 'Exporting VM info for subsequent steps ...'
mapfile -t vm_info < <(get_vm_info "${vm_name}")
vm_id="${vm_info[0]}"
ssh_dest="${vm_info[1]}"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"

echo 'Setting stop scheduler ...'
stop_time="$(jq '.SchedulerStopTime // empty' <<< "${requirements}")"
api_call_retry update_stop_scheduler "${vm_id}" "${stop_time}" > /dev/null

unset COMPSHARE_PRIVATE_KEY

wait_for_vm_to_be_available "${ssh_dest}"
setup_ssh_key "${ssh_dest}" "${encoded_password_file}"
rm -rf "${encoded_password_file}"


#--------------------------------------------------------------------
# Basic setup on the VM.
#--------------------------------------------------------------------
echo 'Uploading tools to the VM ...'
scp -qr -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
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
        "${ssh_dest}" "\
            export VM_DOCKER_MIRROR_URL='${VM_DOCKER_MIRROR_URL:-}'; \
            export VM_HTTP_PROXY='${VM_HTTP_PROXY:-}'; \
            export VM_HTTPS_PROXY='${VM_HTTPS_PROXY:-}'; \
            export VM_PROXY_CERTIFICATE='${VM_PROXY_CERTIFICATE:-}'; \
            bash ./${script}"

    if [[ ${reboot_required[${index}]} == true ]]; then
        echo 'Rebooting for setup to take effect ...'
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}" "sudo reboot" || true
        wait_for_vm_to_be_available "${ssh_dest}"
    fi
done
