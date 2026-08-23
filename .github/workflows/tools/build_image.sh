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
# The following environment variables may need to be set:
# * CLOUD_VENDER
# * SSH_DEST
# * VM_HTTP_PROXY
# * VM_HTTPS_PROXY
# * VM_PIP_INDEX_URL
# * VM_PROXY_CERTIFICATE
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

config_file=''
if [[ -n ${CLOUD_VENDER:-} ]]; then
    config_file="${script_dir}/${CLOUD_VENDER@L}/config.yml"
fi

# Utility functions
script_utils="${script_dir}/utils.sh"

if [[ ${test_group} == *gpu* ]]; then
    compute='gpu'
    extras='[dev,gpu,spark]'
else
    compute='cpu'
    extras='[dev]'
fi

docker_args="-t ${image_tag} \
    -f ${dockerfile} \
    --build-arg COMPUTE=${compute} \
    --build-arg EXTRAS=${extras} \
    --build-arg GIT_REF= \
    --build-arg PYTHON_VERSION=${python_version}"

if [[ -z ${SSH_DEST} ]]; then
    echo 'Building Docker image on current GitHub-hosted runner ...'
    docker build . ${docker_args}
else
    echo "Building Docker image on the newly created VM ..."
    echo '* Copying files to avoid download failure on the VM ...'

    echo '  + Dowloading SDKMan files ...'
    sdkman_sh='sdkman.sh'
    sdkman_zip='sdkman.zip'
    sdkman_native_zip='sdkman-native.zip'
    curl -LsSf -o "${sdkman_sh}" https://get.sdkman.io/?ci=true
    sdkman_service="$(grep 'SDKMAN_SERVICE=' "${sdkman_sh}" | cut -d '"' -f 2)/broker/download"
    sdkman_version="install/$(grep 'SDKMAN_VERSION=' "${sdkman_sh}" | cut -d '"' -f 2)/linuxx64"
    sdkman_native_version="install/$(grep 'SDKMAN_NATIVE_VERSION=' "${sdkman_sh}" | cut -d '"' -f 2)/linuxx64"
    curl -LsSf -o "${sdkman_zip}" "${sdkman_service}/sdkman/${sdkman_version}"
    curl -LsSf -o "${sdkman_native_zip}" "${sdkman_service}/native/${sdkman_native_version}"

    echo '  + Configuring SDKMan in Dockerfile ...'
    sed -i \
        -e "/USER/a \
            WORKDIR /root \
            COPY ./${sdkman_zip} /root/${sdkman_zip} \
            COPY ./${sdkman_native_zip} /root/${sdkman_native_zip}" \
        -e  "/get.sdkman.io/a \
            && sed -i -e \"/* Downloading/a if [[ ! -f /root/${sdkman_zip} && ! -f /root/${sdkman_native_zip} ]]; then\" \\\\\\
                    -e \"/download\\\/sdkman/a else cp /root/${sdkman_zip} \\\\\"\\\\$\{sdkman_zip_file\}\\\\\"; fi\" \\\\\\
                    -e \"/download\\\/native/a else cp /root/${sdkman_native_zip} \\\\\"\\\\$\{sdkman_zip_file\}\\\\\"; fi\" \\\\\\
                    ${sdkman_sh} \\\\" \
        "${dockerfile}"

    echo '  + Configuring APT in Dockerfile ...'
    if [[ -n ${config_file} ]]; then
        apt_mirror="$(yq '.apt_mirror // ""' "${config_file}")"
        if [[ -n ${apt_mirror} ]]; then
            sed -i "/SHELL /a \
                RUN sed -i -e \"s#archive.ubuntu.com#${apt_mirror}#g\" \\\\\\
                        -e \"s#security.ubuntu.com#${apt_mirror}#g\" \\\\\\
                        /etc/apt/sources.list.d/ubuntu.sources" \
                "${dockerfile}"
        fi
    fi

    echo '  + Uploading recommenders ...'
    tar cf ../recommenders.tar ./*
    scp -q -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        ../recommenders.tar "${SSH_DEST}":
    rm -rf ../recommenders.tar
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${SSH_DEST}" "\
            mkdir recommenders \
            && tar xf recommenders.tar -C recommenders \
            && rm -rf recommenders.tar"

    docker_args="${docker_args} \
        --build-arg UV_INSECURE_HOST='github.com'"

    if [[ -n ${VM_HTTP_PROXY:-} || -n ${VM_HTTPS_PROXY:-} ]]; then
        pip_index_ip="$(echo "${VM_PIP_INDEX_URL:-}" \
            | sed -e 's|^.*://||' -e 's|:.*$||')"
        pip_index_ip="${pip_index_ip:+,$pip_index_ip}"
        docker_no_proxy="developer.download.nvidia.com${pip_index_ip}"
        docker_args="${docker_args} \
            --build-arg NO_PROXY='${docker_no_proxy}' \
            --build-arg no_proxy='${docker_no_proxy}'"

        if [[ -n ${VM_HTTP_PROXY:-} ]]; then
            docker_args="${docker_args} \
                --build-arg HTTP_PROXY='${VM_HTTP_PROXY}' \
                --build-arg http_proxy='${VM_HTTP_PROXY}'"
        fi

        if [[ -n ${VM_HTTPS_PROXY:-} ]]; then
            docker_args="${docker_args} \
                --build-arg HTTPS_PROXY='${VM_HTTPS_PROXY}' \
                --build-arg https_proxy='${VM_HTTPS_PROXY}'"
        fi
    fi

    if [[ -n ${VM_PIP_INDEX_URL:-} ]]; then
        docker_args="${docker_args} \
            --build-arg VM_PIP_INDEX_URL='${VM_PIP_INDEX_URL}'"
    fi

    if [[ -n ${VM_PROXY_CERTIFICATE:-} ]]; then
        docker_args="${docker_args} \
            --build-arg VM_PROXY_CERTIFICATE='${VM_PROXY_CERTIFICATE}'"
    fi

    echo '* Building the final image ...'
    source "${script_utils}"
    run_cmd_retry ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=10 \
        "${SSH_DEST}" "\
            cd recommenders; \
            docker build . ${docker_args}"
fi
