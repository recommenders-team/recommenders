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
# * CLOUD_SERVICE_SECRET
# * CLOUD_SERVICE_EXTRA_DATA
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

COMPSHARE_PUBLIC_KEY="$(jq -r '.COMPSHARE_PUBLIC_KEY' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"
VM_DOCKER_MIRROR_URL="$(jq -r '.VM_DOCKER_MIRROR_URL // ""' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"
VM_HTTP_PROXY="$(jq -r '.VM_HTTP_PROXY // ""' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"
VM_HTTPS_PROXY="$(jq -r '.VM_HTTPS_PROXY // ""' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"
VM_PROXY_CERTIFICATE="$(jq -r '.VM_PROXY_CERTIFICATE // ""' \
    <<< "${CLOUD_SERVICE_EXTRA_DATA}")"

export COMPSHARE_PRIVATE_KEY="${CLOUD_SERVICE_SECRET}"
export COMPSHARE_PUBLIC_KEY
export VM_DOCKER_MIRROR_URL
export VM_HTTP_PROXY
export VM_HTTPS_PROXY
export VM_PROXY_CERTIFICATE

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
