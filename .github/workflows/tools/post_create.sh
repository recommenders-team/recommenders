#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Do post-create setup on the VM.
# 
# Params:
# * SSH destination
#   + in the format like username@ip_address
#   + If not set, then no post create actions are performed.
#
# It is assumed there are 3 files in the script directory:
# * configure.sh
#   + Basic configurations, such as
#     - APT mirror and auto update
#     - betwork: HTTP/HTTPS proxies, DNS
# * install_docker.sh
#   + For Docker installation
# * install_nvidia_tools.sh
#   + For installing NVIDIA tools such as CUDA driver and container
#     toolkit
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * CLOUD_SERVICE_ENVS
#   + It contains the following keys in JSON:
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

ssh_dest="${1:-}"

cloud_service="${CLOUD_SERVICE:-}"
cloud_service="${cloud_service@L}"
utils_sh="${script_dir}/utils.sh"

# Setup scripts for configuring network,
# installing Docker and NVIDIA container toolkit
script_dir_name="$(realpath "${script_dir}")"
script_dir_name="${script_dir_name##*/}"
setup_scripts=("${script_dir_name}/configure.sh" \
              "${script_dir_name}/install_docker.sh" \
              "${script_dir_name}/install_nvidia_tools.sh")


#--------------------------------------------------------------------
if [[ -n ${ssh_dest} ]]; then
    echo 'Importing utility functions ...'
    source "${utils_sh}"

    echo 'Uploading tools to the VM ...'
    temp_script_dir="$(mktemp -d)"
    trap 'rm -rf "${temp_script_dir}"' EXIT

    script_dir_tar="${temp_script_dir}/${script_dir_name}.tar"
    tar -cf "${script_dir_tar}" \
        -C "$(dirname "${script_dir}")" \
        --exclude='tf' "${script_dir_name}"
    scp -q -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${script_dir_tar}" "${ssh_dest}":
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}" "tar xf ${script_dir_tar##*/} \
            && rm -rf ${script_dir_tar##*/}"

    for index in "${!setup_scripts[@]}"; do
        script="${setup_scripts[${index}]}"

        wait_for_vm_to_be_available "${ssh_dest}"
        echo "Running ${script} on the VM ..."
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}" "\
                export CLOUD_SERVICE='${cloud_service}'; \
                $(get_env_exports "${CLOUD_SERVICE_ENVS:-}") \
                bash ./${script}"
    done

    echo 'Rebooting for setup to take effect ...'
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}" "rm -rf ${script_dir_name} && sudo reboot" \
    || true
    wait_for_vm_to_be_available "${ssh_dest}"
fi
