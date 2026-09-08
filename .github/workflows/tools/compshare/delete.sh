#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Delete a CompShare VM
# 
# Params:
# * VM name
#
# The following environment variables must be set:
# * CLOUD_SERVICE_SECRET
#   + It contains the private key for CompShare APIs and is used as
#     COMPSHARE_PRIVATE_KEY in the script.
# * CLOUD_SERVICE_ENVS
#   + It contains the following keys in JSON:
#     - COMPSHARE_PUBLIC_KEY (required)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

vm_name="${1:-}"
[[ -z ${vm_name} ]] && { echo 'No VM specified.'; exit 0; }


#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "$(dirname "$0")/utils.sh"

echo 'Exporting environment variables ...'
export COMPSHARE_PRIVATE_KEY="${CLOUD_SERVICE_SECRET}"
eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"

delay=5
num_attempts=6
attempt=1
while true; do
    mapfile -t vm_info < <(get_vm_info "${vm_name}")
    if [[ -n ${vm_info:-} ]]; then
        vm_id="${vm_info[0]}"
        echo "Stopping the VM ${vm_name} ..."
        api_call_retry stop_instance "${vm_id}" > /dev/null

        echo "Deleting the VM ${vm_name} ..."
        api_call_retry 10 terminate_instance "${vm_id}" > /dev/null
        break
    fi

    if (( attempt >= num_attempts )); then
        echo "The VM ${vm_name} may not be created."
        exit 0
    fi
    echo "Attempt ${attempt} failed! The VM info may not be available. Retrying in ${delay} seconds ..." >&2
    sleep "${delay}"
    ((attempt++))
done
