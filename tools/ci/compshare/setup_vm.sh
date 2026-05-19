#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create a CompShare VM and prepare the environment for testing
# 
# Params:
# * VM name
#
# The following environment variables must be set:
# * COMPSHARE_PRIVATE_KEY
# * COMPSHARE_PUBLIC_KEY
# * DOCKER_MIRROR_URL
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

# Setup scripts for configuring network,
# installing Docker and NVIDIA container toolkit
SCRIPT_SETUP=("${SCRIPT_DIR}/configure.sh" \
              "${SCRIPT_DIR}/install_docker.sh" \
              "${SCRIPT_DIR}/install_container_toolkit.sh")

# Indicators for whether reboot is required after running each setup script
REBOOT_REQUERED=('yes' 'no' 'yes')

echo 'Importing utility functions ...'
source "${SCRIPT_UTILS}"

encoded_password_file="$(mktemp)"
mktemp -u XXXXXXXXXX | tr -d '\n' | base64 | tr -d '\n' > "${encoded_password_file}"
allocate_vm "${vm_name}" "${encoded_password_file}"
mapfile -t vm_info < <(get_vm_info "${vm_name}")
ssh_dest="${vm_info[1]}"
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
    script="${SCRIPT_SETUP[${index}]##*/}"

    wait_for_vm_to_be_available "${ssh_dest}"
    echo "Running ${script} on the VM ..."
    ssh -t -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null \
      "${ssh_dest}" "export DOCKER_MIRROR_URL='${DOCKER_MIRROR_URL:-}'; bash ./${script}"

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
