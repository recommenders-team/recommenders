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

SCRIPT_DIR="$(dirname "$0")"
vm_name="${1:-}"
more_than_half_hour="${2:-}"
requirements="${3:-}"
[[ -z "${vm_name}" || -z "${more_than_half_hour}" ]] && exit 1

# CompShare API specification JSON file
COMPSHARE_SPEC_FILE="${SCRIPT_DIR}/spec.json"

# Utility functions
SCRIPT_UTILS="${SCRIPT_DIR}/utils.sh"

# Setup scripts for configuring network,
# installing Docker and NVIDIA container toolkit
SCRIPT_SETUP=("${SCRIPT_DIR}/configure.sh" \
              "${SCRIPT_DIR}/install_docker.sh" \
              "${SCRIPT_DIR}/install_nvidia_tools.sh")

# Indicators for whether reboot is required after running each setup script
REBOOT_REQUERED=('yes' 'no' 'yes')

echo 'Importing utility functions ...'
source "${SCRIPT_UTILS}"

encoded_password_file="$(mktemp)"
mktemp -u XXXXXXXXXX | tr -d '\n' | base64 | tr -d '\n' > "${encoded_password_file}"
allocate_vm "${vm_name}" "${encoded_password_file}" "${more_than_half_hour}" "${requirements}"
mapfile -t vm_info < <(get_vm_info "${vm_name}")
vm_id="${vm_info[0]}"
ssh_dest="${vm_info[1]}"

echo "Setting stop scheduler ..."
count=0
while true; do
    response=$(update_stop_scheduler "${vm_id}")
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
unset COMPSHARE_PRIVATE_KEY
unset COMPSHARE_PUBLIC_KEY

wait_for_vm_to_be_available "${ssh_dest}"
setup_ssh_key "${ssh_dest}" "${encoded_password_file}"
rm -rf "${encoded_password_file}"

echo "Uploading scripts to the VM ..."
for script in "${SCRIPT_SETUP[@]}"; do
    echo "* ${script}"
    scp -q -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${script}" "${ssh_dest}":
done

for index in "${!SCRIPT_SETUP[@]}"; do
    script="$(basename "${SCRIPT_SETUP[${index}]}")"

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

    echo "Removing ${script} ..."
    ssh -t -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      "${ssh_dest}" "rm -rf ./${script}"

    if [[ "${REBOOT_REQUERED[${index}]}" == 'yes' ]]; then
        echo 'Rebooting for setup to take effect ...'
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}" "sudo reboot" || true
        wait_for_vm_to_be_available "${ssh_dest}"
    fi
done
