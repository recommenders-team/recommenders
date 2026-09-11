#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Build the Docker image
#
# Params:
# * Docker image tag
# * Path to Dockerfile in the repo
# * Test group
# * Python version
#
# It assumes that there is a configuration file called
#
#     config.yml
#
# in a directory named '${CLOUD_SERVICE@L}' under the
# script directory.  In config.yml, the following key may need to be
# set:
# * apt_mirror
#   + the APT mirror to be used for downloading system packages.
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * CLOUD_SERVICE_ENVS
#   + It contains the following keys in JSON:
#     - VM_HTTP_PROXY (optional)
#     - VM_HTTPS_PROXY (optional)
#     - VM_PIP_INDEX_URL (optional)
#     - VM_PROXY_CERTIFICATE (optional)
# * SSH_DEST
#   + in the format like username@ip_address
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
image_tag="${1:-}"
dockerfile="${2:-}"
test_group="${3:-}"
python_version="${4:-}"

[[ -z ${image_tag} \
  || -z ${dockerfile} \
  || -z ${test_group} \
  || -z ${python_version} ]] && exit 1

cloud_service="${CLOUD_SERVICE:-}"
cloud_service="${cloud_service@L}"
config_yml="${script_dir}/${cloud_service}/config.yml"
recommenders_dir_name='recommenders'
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
# Determine build args
#--------------------------------------------------------------------
if [[ ${test_group} == *gpu* ]]; then
    compute='gpu'
    extras_gpu=',gpu'
else
    compute='cpu'
    extras_gpu=''
fi

if [[ ${test_group} == *spark* ]]; then
    extras_spark=',spark'
else
    extras_spark=''
fi

docker_args=(\
    -t "${image_tag}" \
    -f "${dockerfile}" \
    --build-arg "COMPUTE=${compute}" \
    --build-arg "EXTRAS=[dev${extras_gpu}${extras_spark}]" \
    --build-arg 'GIT_REF=' \
    --build-arg "PYTHON_VERSION=${python_version}")


#--------------------------------------------------------------------
# Determine the VM for Docker build
#--------------------------------------------------------------------
if [[ -z ${SSH_DEST:-} ]]; then
    echo 'Building Docker image on current GitHub-hosted runner ...'
    docker build . "${docker_args[@]}"
else
    echo 'Importing utility functions ...'
    source "${utils_sh}"

    echo 'Exporting environment variables ...'
    eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"

    pre_image_build "${SSH_DEST}" "${dockerfile}" "${config_yml}" \
        "${recommenders_dir_name}"

    echo '* Building the Docker image on the VM ...'
    docker_args+=(--build-arg 'UV_INSECURE_HOST=github.com')

    if [[ -n ${VM_HTTP_PROXY:-} || -n ${VM_HTTPS_PROXY:-} ]]; then
        pip_index_ip="$(echo "${VM_PIP_INDEX_URL:-}" \
            | sed -e 's|^.*://||' -e 's|:.*$||')"
        pip_index_ip="${pip_index_ip:+,$pip_index_ip}"
        docker_no_proxy="developer.download.nvidia.com${pip_index_ip}"
        docker_args+=(\
            --build-arg "NO_PROXY=${docker_no_proxy}" \
            --build-arg "no_proxy=${docker_no_proxy}")

        if [[ -n ${VM_HTTP_PROXY:-} ]]; then
            docker_args+=(\
                --build-arg "HTTP_PROXY=${VM_HTTP_PROXY}" \
                --build-arg "http_proxy=${VM_HTTP_PROXY}")
        fi

        if [[ -n ${VM_HTTPS_PROXY:-} ]]; then
            docker_args+=(\
                --build-arg "HTTPS_PROXY=${VM_HTTPS_PROXY}" \
                --build-arg "https_proxy=${VM_HTTPS_PROXY}")
        fi
    fi

    if [[ -n ${VM_PIP_INDEX_URL:-} ]]; then
        docker_args+=(--build-arg "VM_PIP_INDEX_URL=${VM_PIP_INDEX_URL}")
    fi

    if [[ -n ${VM_PROXY_CERTIFICATE:-} ]]; then
        docker_args+=(--build-arg "VM_PROXY_CERTIFICATE=${VM_PROXY_CERTIFICATE}")
    fi

    run_cmd_retry ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=10 \
        "${SSH_DEST}" "\
            cd ${recommenders_dir_name} \
            && docker build ." "${docker_args[@]}"
fi
