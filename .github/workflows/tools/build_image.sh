#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Build the Docker image
#
# Params:
# * SSH destination
#   + in the format like username@ip_address
#   + If not set, then the Docker image will be built on the current
#     machine.
# * Name for the Docker image tag and the code directory
# * Path to Dockerfile in the repo
# * Test group
# * Python version
# * Path to the virtual environment created inside the image
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
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
ssh_dest="${1:-}"
unique_name="${2:-}"
dockerfile="${3:-}"
test_group="${4:-}"
python_version="${5:-}"
venv_dir="${6:-}"

[[ -z ${unique_name} \
  || -z ${dockerfile} \
  || -z ${test_group} \
  || -z ${python_version} \
  || -z ${venv_dir} ]] && echo 'Parameter error!' >&2 && exit 1

cloud_service="${CLOUD_SERVICE:-}"
cloud_service="${cloud_service@L}"
config_yml="${script_dir}/${cloud_service}/config.yml"
image_tag="${unique_name}"
recommenders_dir_name="${unique_name}"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
# Generate basic Docker build args
#--------------------------------------------------------------------
echo 'Generating basic Docker build args ...'
if [[ ${test_group} == *gpu* ]]; then
    compute='gpu'
    extras_gpu='gpu'
else
    compute='cpu'
fi

if [[ ${test_group} == *spark* ]]; then
    extras_spark='spark'
fi

docker_args=(\
    -t "${image_tag}" \
    -f "${dockerfile}" \
    --build-arg "COMPUTE=${compute}" \
    --build-arg "EXTRAS=[dev${extras_gpu:+,$extras_gpu}${extras_spark:+,$extras_spark}]" \
    --build-arg 'GIT_REF=' \
    --build-arg "PYTHON_VERSION=${python_version}" \
    --build-arg "VENV_DIR=${venv_dir}")


#--------------------------------------------------------------------
# Generate extra Docker build args
#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "${utils_sh}"

echo 'Exporting environment variables ...'
eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"

echo 'Generating extra Docker build args ...'
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
        if [[ -z ${VM_PROXY_CERTIFICATE:-} ]]; then
            echo 'VM_HTTPS_PROXY set but VM_PROXY_CERTIFICATE not!' >&2
            exit 1
        fi
        docker_args+=(\
            --build-arg "HTTPS_PROXY=${VM_HTTPS_PROXY}" \
            --build-arg "https_proxy=${VM_HTTPS_PROXY}" \
            --build-arg "VM_PROXY_CERTIFICATE=${VM_PROXY_CERTIFICATE}" \
            --build-arg 'UV_INSECURE_HOST=github.com')
    fi
fi

if [[ -n ${VM_PIP_INDEX_URL:-} ]]; then
    docker_args+=(--build-arg "VM_PIP_INDEX_URL=${VM_PIP_INDEX_URL}")
fi


#--------------------------------------------------------------------
# Build Docker image
#--------------------------------------------------------------------
if [[ -z ${ssh_dest} ]]; then
    echo 'Building the Docker image on current runner ...'
    docker build . "${docker_args[@]}"
else
    pre_image_build "${ssh_dest}" "${dockerfile}" "${config_yml}" \
        "${recommenders_dir_name}"

    echo 'Building the Docker image on the VM ...'
    mapfile -t docker_args < <(printf '%q\n' "${docker_args[@]}")
    run_cmd_retry ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=10 \
        "${ssh_dest}" "\
            cd ${recommenders_dir_name} \
            && docker build ." "${docker_args[@]}"
fi
