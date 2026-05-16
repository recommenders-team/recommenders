#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Install Docker in rootless mode
#
# The following environment variables may need to be set:
# * VM_DOCKER_MIRROR_URL
# * VM_HTTP_PROXY
# * VM_HTTPS_PROXY
# * VM_PIP_INDEX_URL
#
# See
# * https://docs.docker.com/engine/install/ubuntu/
# * https://docs.docker.com/engine/security/rootless/
######################################################################
set -euo pipefail
shopt -s inherit_errexit

ARCH="$(dpkg --print-architecture)"
CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
APT_URL="https://download.docker.com/linux/ubuntu"
APT_LIST="/etc/apt/sources.list.d/docker.list"
KEYRING_DIR="/etc/apt/keyrings"
GPG_PATH="${KEYRING_DIR}/docker.asc"
GPG_URL="${APT_URL}/gpg"
APT_ENTRY="deb [arch=${ARCH} signed-by=${GPG_PATH}] ${APT_URL} ${CODENAME} stable"

echo '* Installing prerequisites ...'
while apt_lock_pid=$(sudo fuser /var/lib/apt/lists/lock 2>/dev/null); do
    echo '    - Releasing /var/lib/apt/lists/lock ...'
    sudo kill "${apt_lock_pid}"
    sleep 5
done
sudo apt-get update
sudo dpkg --configure -a
count=0
until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
    apt-get install -y ca-certificates curl jq; do
    echo '  + Failed to install.'
    count=$((count + 1))
    if [[ $count -lt 5 ]]; then
        sleep 5
        echo '  + Trying again ...'
    else
        exit 1
    fi
done

echo '* Adding Docker official GPG key ...'
sudo install -m 0755 -d "${KEYRING_DIR}"
sudo curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors "${GPG_URL}" -o "${GPG_PATH}"
sudo chmod a+r "${GPG_PATH}"

echo '* Setting APT repo source for Docker ...'
sudo mkdir -p "$(dirname "${APT_LIST}")"
echo "${APT_ENTRY}" | sudo tee "${APT_LIST}" > /dev/null
sudo apt-get update

echo '* Installing the latest Docker community edition ...'
count=0
until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
    apt-get install -y docker-ce; do
    echo '  + Failed to install.'
    count=$((count + 1))
    if [[ $count -lt 5 ]]; then
        sleep 5
        echo '  + Trying again ...'
    else
        exit 1
    fi
done

echo '* Configuring Docker daemon in rootless mode ...'
echo '  - Installing prerequisites ...'
count=0
until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
    apt-get install -y uidmap docker-ce-rootless-extras; do
    echo '  + Failed to install.'
    count=$((count + 1))
    if [[ $count -lt 5 ]]; then
        sleep 5
        echo '  + Trying again ...'
    else
        exit 1
    fi
done

echo '  - Disabling system-wide Docker daemon ...'
sudo systemctl disable --now docker.service docker.socket
sudo rm /var/run/docker.sock

echo '  - Installing rootless Docker daemon ...'
dockerd-rootless-setuptool.sh install

if [[ -n "${VM_DOCKER_MIRROR_URL:-}" \
    || -n "${VM_HTTP_PROXY:-}" \
    || -n "${VM_HTTPS_PROXY:-}" \
    || -n "${VM_PIP_INDEX_URL:-}" ]]; then
    echo '* Configuring proxies for Docker ...'
    update_json_config() {
        local json_file="${1:-}"
        local updates="${2:-}"
        [[ -z "${updates}" || -z "${json_file}" ]] && return 1

        if [[ -f "${json_file}" ]]; then
            echo "    ## Updating ${json_file} ..."
            local temp_json
            temp_json=$(jq -s '
                def update($a; $b):
                    ($a | type) as $ta | ($b | type) as $tb |
                    if $ta == "object" and $tb == "object" then
                        reduce ([$a, $b] | add | keys_unsorted[]) as $k ({};
                        .[$k] = update($a[$k]; $b[$k]))
                    elif $ta == "array" and $tb == "array" then
                        $a + $b
                    else
                        $b // $a
                    end;
                reduce .[] as $item (null; update(.; $item))' \
                "${json_file}" <(echo "${updates}"))
            echo "${temp_json}" > "${json_file}"
        else
            echo "    ## Creating ${json_file} ..."
            mkdir -p "$(dirname "${json_file}")"
            echo "${updates}" | jq '.' > "${json_file}"
        fi
    }

    if [[ -n "${VM_DOCKER_MIRROR_URL:-}" ]]; then
        echo '  + Setting Docker mirror URL ...'
        update_json_config \
            "${HOME}/.config/docker/daemon.json" \
            "{ \"registry-mirrors\": [ \"${VM_DOCKER_MIRROR_URL}\" ] }"
    fi

    docker_config_json="${HOME}/.docker/config.json"
    if [[ -n "${VM_HTTP_PROXY:-}" ]]; then
        echo '  + Setting HTTP proxy for docker build and docker run ...'
        update_json_config "${docker_config_json}" \
            "{ \"proxies\": { \"default\": { \"httpProxy\": \"${VM_HTTP_PROXY}\" } } }"
    fi

    if [[ -n "${VM_HTTPS_PROXY:-}" ]]; then
        echo '  + Setting HTTP proxy for docker build and docker run ...'
        update_json_config "${docker_config_json}" \
            "{ \"proxies\": { \"default\": { \"httpsProxy\": \"${VM_HTTPS_PROXY}\" } } }"
    fi

    echo '  + Setting no proxy for pip index ...'
    pip_index_ip="$(echo "${VM_PIP_INDEX_URL:-}" | sed -e 's|^.*://||' -e 's|:.*$||')"
    update_json_config "${docker_config_json}" \
        "{ \"proxies\": { \"default\": { \"noProxy\": \"${pip_index_ip}\" } } }"
fi

echo '* Starting rootless Docker daemon ...'
systemctl --user start docker

echo '* Enabling Docker service and launch the daemon on startup ...'
systemctl --user enable docker
sudo loginctl enable-linger "$(whoami)"
