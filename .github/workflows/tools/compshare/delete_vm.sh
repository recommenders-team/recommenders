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

vm_name="${1:-}"
[[ -z "${vm_name}" ]] && exit 1

echo 'Importing utility functions ...'
source "$(dirname "$0")/utils.sh"

mapfile -t vm_info < <(get_vm_info "${vm_name}")
if [[ -n "${vm_info:-}" ]]; then
    vm_id="${vm_info[0]}"
    echo "Stopping the VM ${vm_name} ..."
    api_call_retry stop_instance "${vm_id}" > /dev/null

    echo "Deleting the VM ${vm_name} ..."
    api_call_retry terminate_instance "${vm_id}" > /dev/null
fi
