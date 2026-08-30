#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Do post-create setup on the VM.
#
# It assumed that there is a configuration file called
# config.yml in a directory named '${CLOUD_SERVICE@L}' under the
# script directory.  In config.yml,
# * there is a key 'scripts.post_create' with the list of scripts
#   for post-create setup as its value.
# * Each item of the script list is an object with a key 'script'
#   referring to the path to the script and a key 'reboot'
#   indicating whether reboot is required after running the
#   script.
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE
#   + It should be the name of parent directory.
# * CLOUD_SERVICE_EXTRA_DATA
#   + It contains the following keys in JSON:
#     - VM_DOCKER_MIRROR_URL (optional)
#       * Semicolon separated URLs of docker mirrors
#     - VM_HTTP_PROXY (optional)
#     - VM_HTTPS_PROXY (optional)
#     - VM_PIP_INDEX_URL (optional)
#     - VM_PROXY_CERTIFICATE (optional)
# * SSH_DEST
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
cloud_service="${CLOUD_SERVICE:-}"
cloud_service="${cloud_service@L}"
config_yml="${script_dir}/${cloud_service}/config.yml"

if [[ -n ${SSH_DEST} && -f ${config_yml} ]] \
    && jq -e '.scripts.post_create' "${config_yml}" 2>/dev/null; then

    echo 'Uploading tools to the VM ...' >&2
    scp -qr -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${script_dir}" "${SSH_DEST}":

    readarray -d '' post_create_scripts < \
        <(yq -0 '.scripts.post_create[].script' "${config_yml}")
    readarray -d '' reboot_required < \
        <(yq -0 '.scripts.post_create[].reboot' "${config_yml}")
    service_tools_dir="${script_dir##*/}/${cloud_service}"
    for index in "${!post_create_scripts[@]}"; do
        script="${service_tools_dir}/${post_create_scripts[${index}]}"

        wait_for_vm_to_be_available "${SSH_DEST}"
        echo "Running ${script} on the VM ..." >&2
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${SSH_DEST}" "\
                $(jq -r '[to_entries | .[] | "export \(.key)=\(.value | @sh);"] | join(" ")' \
                    <<< "${CLOUD_SERVICE_EXTRA_DATA:-}") \
                bash ./${script}"

        if [[ ${reboot_required[${index}]} == 'true' ]]; then
            echo 'Rebooting for setup to take effect ...' >&2
            ssh -t -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                "${SSH_DEST}" "sudo reboot" || true
            wait_for_vm_to_be_available "${SSH_DEST}"
        fi
    done
fi
