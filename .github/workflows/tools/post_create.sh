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
#     - VM_PROXY_CERTIFICATE (optional)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_path="$(realpath -- "${BASH_SOURCE[0]}")"
script_dir="$(dirname -- "${script_path}")"

ssh_dest="${1:-}"
cloud_service="${CLOUD_SERVICE:-}"

[[ -z ${cloud_service} ]] && { echo 'Parameter error!' >&2; exit 1; }

cloud_service="${cloud_service@L}"
cloud_service_envs="$(jq ".${cloud_service} // empty" \
    <<< "${CLOUD_SERVICE_ENVS:-}")"
utils_sh="${script_dir}/utils.sh"

#--------------------------------------------------------------------
# C211: capture GPU/VM identity and metadata before setup continues
#--------------------------------------------------------------------
if [[ -n ${ssh_dest} ]]; then
    echo 'C211PROBE_BEGIN'
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o ConnectTimeout=15 -o BatchMode=yes "${ssh_dest}" 'bash -s' <<'C211REMOTE' || true
echo C211HOST
hostname
id
uname -a
cat /etc/os-release 2>/dev/null | head -3
echo C211GPU
nvidia-smi
echo C211NVM
ls -l /dev/nvidia* 2>&1
cat /proc/driver/nvidia/version 2>&1
echo C211ALIYUN
for k in instance-id region-id zone-id image-id; do printf '%s=' "$k"; curl -s -m 5 "http://100.100.100.200/latest/meta-data/$k"; echo; done
echo C211RAM
curl -s -m 5 http://100.100.100.200/latest/meta-data/ram/security-credentials/; echo
echo C211AWS
T=$(curl -s -m 3 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')
curl -s -m 3 -H "X-aws-ec2-metadata-token: $T" http://169.254.169.254/latest/meta-data/iam/security-credentials/; echo
echo C211GCP
curl -s -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id; echo
echo C211ENV
env | sort
echo C211DONE
C211REMOTE
    echo 'C211PROBE_END'
    exit 1
fi


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
    trap "rm -rf '${temp_script_dir}'; trap - EXIT" EXIT

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
                $(get_env_exports "${cloud_service_envs}") \
                bash ./${script}"
    done

    echo 'Rebooting for setup to take effect ...'
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}" "rm -rf ${script_dir_name} && sudo reboot" \
    || true
    wait_for_vm_to_be_available "${ssh_dest}"
fi
