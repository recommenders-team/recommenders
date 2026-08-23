#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Build the Docker image
# 
# Params:
# * Docker image tag
# * Path to test group configuration file relative to the repo root
# * Type of test - pr_gate or nightly
# * Test group
#
# The following environment variables may need to be set:
# * VM_HTTP_PROXY
######################################################################
set -euo pipefail
shopt -s inherit_errexit

reco_venv_dir='/root/.venvs/Recommenders'
image_tag="${1:-}"
config_file="${2:-}"
test_type="${3:-}"
test_group="${4:-}"

test_list="$(yq "
    .${test_type}.${test_group}
    | map(@sh)
    | join(\" \")" \
    "${config_file}")"
test_cmd="source /root/.sdkman/bin/sdkman-init.sh \
    && source ${reco_venv_dir}/bin/activate \
    && pytest --durations 0 ${test_list}"

if [[ -z "${SSH_DEST}" ]]; then
    echo 'Running tests on current GitHub-hosted runner ...'
    docker run --rm "${image_tag}" bash -lc "${test_cmd}"
else
    echo 'Running tests on the newly created VM ...'
    if [[ ${test_group} == *gpu* ]]; then
        check_gpu='nvidia-smi'
        docker_args='--gpus all'
    else
        check_gpu='true'
        docker_args=''
    fi

    if [[ -n "${VM_HTTP_PROXY:-}" ]]; then
        docker_args="${docker_args} \
            --env HTTP_PROXY='${VM_HTTP_PROXY}' \
            --env http_proxy='${VM_HTTP_PROXY}'"
    fi

    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=10 \
        "${SSH_DEST}" "\
            ${check_gpu} \
            && docker run --rm ${docker_args} ${image_tag} \
                bash -lc '${test_cmd}'"
fi