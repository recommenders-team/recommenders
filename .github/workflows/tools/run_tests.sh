#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Run the tests in a Docker container.
# 
# Params:
# * SSH destination
#   + in the format like username@ip_address
#   + If not set, then the tests are run on the current machine.
# * Docker image tag
# * Path to test group configuration file relative to the repo root
# * Type of test - pr_gate or nightly
# * Test group
# * Path to the virtual environment created inside the image
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE_ENVS
#   + It contains the following keys in JSON:
#     - VM_HTTP_PROXY (optional)
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"
ssh_dest="${1:-}"
image_tag="${2:-}"
test_groups_yml="${3:-}"
test_type="${4:-}"
test_group="${5:-}"
venv_dir="${6:-}"

[[ -z ${image_tag} \
  || -z ${test_groups_yml} \
  || -z ${test_type} \
  || -z ${test_group} \
  || -z ${venv_dir} ]] && echo 'Parameter error!' >&2 && exit 1

utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
# Get test list
#--------------------------------------------------------------------
test_list="$(yq "
    .${test_type}.${test_group}
    | map(@sh)
    | join(\" \")" \
    "${test_groups_yml}")"
test_cmd="source /root/.sdkman/bin/sdkman-init.sh \
    && source ${venv_dir}/bin/activate \
    && pytest --durations 0 ${test_list}"


#--------------------------------------------------------------------
# Generate Docker run args
#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "${utils_sh}"

echo 'Exporting environment variables ...'
eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"


echo 'Generating Docker run args ...'
docker_args=()
if [[ ${test_group} == *gpu* ]]; then
    check_gpu='nvidia-smi'
    docker_args+=(--gpus all)
else
    check_gpu='true'
fi

if [[ -n ${VM_HTTP_PROXY:-} ]]; then
    docker_args+=(\
        --env "HTTP_PROXY=${VM_HTTP_PROXY}" \
        --env "http_proxy=${VM_HTTP_PROXY}")
fi

#--------------------------------------------------------------------
# Run tests
#--------------------------------------------------------------------
if [[ -z ${ssh_dest} ]]; then
    echo 'Running tests on current GitHub-hosted runner ...'
    "${check_gpu}" \
    && docker run --rm "${docker_args[@]}" "${image_tag}" \
        bash -lc "${test_cmd}"
else
    echo 'Running tests on the VM ...'
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=10 \
        "${ssh_dest}" "\
            ${check_gpu} \
            && docker run --rm" "${docker_args[@]}" "${image_tag} \
                bash -lc '${test_cmd}'"
fi
