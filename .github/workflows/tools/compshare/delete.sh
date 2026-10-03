#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Delete the CompShare VM
# 
# Params:
# * VM name
#
# NOTE:
# * It is assumed that there is a configuration file called
#
#     config.yml
#
#   in a directory named '${CLOUD_SERVICE@L}' under the
#   script directory.  In config.yml, the following key may need to be
#   set:
#   + secret_key_name
#     - the name of the secret of private key for the cloud service
#       indicated by the environment variable CLOUD_SERVICE.
#
# The following environment variables must be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * CLOUD_SERVICE_SECRET
#   + It contains the private key for CompShare APIs and is used as
#     COMPSHARE_PRIVATE_KEY in the script.
# * CLOUD_SERVICE_ENVS
#   + It contains the following keys in JSON:
#     - COMPSHARE_PUBLIC_KEY (required)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_path="$(realpath -- "${BASH_SOURCE[0]}")"
script_dir="$(dirname -- "${script_path}")"
vm_name="${1:-}"

[[ -z ${vm_name} ]] && exit 0

[[ -z ${CLOUD_SERVICE:-} ]] \
    && echo 'CLOUD_SERVICE not set!' >&2 && exit 1

cloud_service="${CLOUD_SERVICE}"
cloud_service="${cloud_service@L}"
config_dir="${script_dir}/${cloud_service}"
config_yml="${config_dir}/config.yml"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "${utils_sh}"

echo 'Exporting environment variables ...'
secret_key_name="$(yq '.secret_key_name' < "${config_yml}")"
export "${secret_key_name}"="${CLOUD_SERVICE_SECRET}"
eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"

delay=5
num_attempts=6
attempt=1
while true; do
    vm_info="$(get_vm_info "${vm_name}" "${config_yml}")"
    if [[ -n ${vm_info:-} ]]; then
        echo "Stopping the VM ${vm_name} ..."
        api_call_retry invoke_action "${config_yml}" \
            'StopCompShareInstance' "${vm_info}" > /dev/null

        wait_for_vm_to_stop "${vm_name}" "${config_yml}"

        echo "Deleting the VM ${vm_name} ..."
        api_call_retry 10 invoke_action "${config_yml}" \
            'TerminateCompShareInstance' "${vm_info}" > /dev/null
        break
    fi

    if (( attempt >= num_attempts )); then
        echo "The VM ${vm_name} may not be created."
        exit 0
    fi
    echo "* Attempt ${attempt} failed! The VM info may not be available. Retrying in ${delay} seconds ..." >&2
    sleep "${delay}"
    ((attempt++))
done
