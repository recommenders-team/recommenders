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
# * COMPSHARE_PRIVATE_KEY
# * COMPSHARE_PUBLIC_KEY
######################################################################
set -euo pipefail
shopt -s inherit_errexit

SCRIPT_DIR="$(dirname "$0")"
vm_name="${1:-}"
[[ -z "${vm_name}" ]] && exit 1

# CompShare API specification JSON file
COMPSHARE_SPEC_FILE="${SCRIPT_DIR}/spec.json"

# Utility functions
SCRIPT_UTILS="${SCRIPT_DIR}/utils.sh"

echo 'Importing utility functions ...'
source "${SCRIPT_UTILS}"

mapfile -t vm_info < <(get_vm_info "${vm_name}")
vm_id="${vm_info[0]}"
if [[ -n "${vm_id}" ]]; then
    echo "Stopping the VM ${vm_name} ..."
    count=0
    while true; do
        response=$(stop_instance "${vm_id}")
        retcode="$(echo "${response}" | jq '.RetCode')"
        if [[ ${retcode} == 0 ]]; then
            break
        fi
        echo "* Failed to stop the VM: ${response}"
        count=$((count + 1))
        if [[ $count -lt 5 ]]; then
            sleep $(( (count+1) * 5 ))
            echo '* Trying again ...'
        else
            exit 1
        fi
    done

    echo "Deleting the VM ${vm_name} ..."
    count=0
    while true; do
        response=$(terminate_instance "${vm_id}")
        retcode="$(echo "${response}" | jq '.RetCode')"
        if [[ ${retcode} == 0 ]]; then
            break
        fi
        echo "* Failed to delete the VM: ${response}"
        count=$((count + 1))
        if [[ $count -lt 5 ]]; then
            sleep $(( (count+1) * 5 ))
            echo '* Trying again ...'
        else
            exit 1
        fi
    done
fi
