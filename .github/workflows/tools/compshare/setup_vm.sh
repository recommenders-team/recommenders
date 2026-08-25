#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Run create_vm.sh to set up a CompShare VM.
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

# Utility functions
script_utils="${script_dir}/utils.sh"
script_create="${script_dir}/create_vm.sh"

export COMPSHARE_PRIVATE_KEY="${CLOUD_SERVICE_SECRET}"
eval "$(jq -r 'to_entries | .[] | "export \(.key)=\(.value | @sh)"' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"

# All VMs from CompShare have GPUs.
# Nightly tests require more time and more GPU memory.
# VMs with Spot ChargeType are cheaper but there is a risk of being
# deleted after 1 hour.
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
bash "${script_create}" "${vm_name}" "${requirements}"

unset COMPSHARE_PRIVATE_KEY

echo 'Exporting VM info for subsequent steps ...'
source "${script_utils}"
mapfile -t vm_info < <(get_vm_info "${vm_name}")
ssh_dest="${vm_info[1]}"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"
