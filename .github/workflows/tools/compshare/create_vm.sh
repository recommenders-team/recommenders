#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create a CompShare VM and prepare the environment for testing
# 
# Params:
# * VM name
# * whether the VM will be used for more than half an hour
#   + Unit tests require less than half hour
#   + Nightly tests require more than half hour
# * (optional) requirements in JSON, for example
#   + {"GPUType":"!2080,P40","Memory":{"GPU":10,"CPU":9}}
#     - It means the GPUType should not be 2080 and P40,
#       GPU memory should be greater than or equal to 10GB
#       and CPU 9GB.
#   + {"GPUType":"2080,P40"}
#     - It means the GPUType should be 2080 or P40.
#
# The following environment variables must be set:
# * CLOUD_SERVICE
#   + It should be the name of parent directory.
# * COMPSHARE_PRIVATE_KEY
# * COMPSHARE_PUBLIC_KEY
#
# The following environment variables may need to be set:
# * VM_DOCKER_MIRROR_URL
# * VM_HTTP_PROXY
# * VM_HTTPS_PROXY
# * VM_PROXY_CERTIFICATE
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
vm_name="${1:-}"
requirements="${2:-}"
[[ -z ${vm_name} ]] && exit 1

tools_dir="${script_dir}/../../tools"
config_file="${script_dir}/config.yml"

# Utility functions
script_utils="${script_dir}/utils.sh"

echo 'Importing utility functions ...'
source "${script_utils}"

encoded_password_file="$(mktemp)"
mktemp -u XXXXXXXXXX | tr -d '\n' | base64 \
    | tr -d '\n' > "${encoded_password_file}"
allocate_vm \
    "${vm_name}" \
    "${encoded_password_file}" \
    "$(jq 'del(.SchedulerStopTime)' <<< "${requirements}")"
mapfile -t vm_info < <(get_vm_info "${vm_name}")
vm_id="${vm_info[0]}"
ssh_dest="${vm_info[1]}"

echo 'Setting stop scheduler ...'
stop_time="$(jq '.SchedulerStopTime // empty' <<< "${requirements}")"
api_call_retry update_stop_scheduler "${vm_id}" "${stop_time}" > /dev/null

unset COMPSHARE_PRIVATE_KEY
unset COMPSHARE_PUBLIC_KEY

wait_for_vm_to_be_available "${ssh_dest}"
setup_ssh_key "${ssh_dest}" "${encoded_password_file}"
rm -rf "${encoded_password_file}"

echo 'Uploading tools to the VM ...'
scp -qr -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "${tools_dir}" "${ssh_dest}":

readarray -d '' post_create_scripts < \
    <(yq -0 '.scripts.post_create[].script' "${config_file}")
readarray -d '' reboot_required < \
    <(yq -0 '.scripts.post_create[].reboot' "${config_file}")
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
