#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Install Docker in rootless mode
#
# The following environment variables must be set:
# * DOCKER_MIRROR_URL
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
sudo apt-get update
sudo dpkg --configure -a
count=0
until sudo apt-get install -y ca-certificates curl; do
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
sudo mkdir -p "${APT_LIST%/*}"
echo "${APT_ENTRY}" | sudo tee "${APT_LIST}" > /dev/null
sudo apt-get update

echo '* Installing the latest Docker community edition ...'
count=0
until sudo apt-get install -y docker-ce; do
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
until sudo apt-get install -y uidmap docker-ce-rootless-extras; do
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

if [[ -n "${DOCKER_MIRROR_URL:-}" ]]; then
    echo '  - Setting Docker mirror URL ...'
    DAEMON_JSON="${HOME}/.config/docker/daemon.json"
    if [[ -f "${DAEMON_JSON}" ]]; then
        echo "    ## Appending to ${DAEMON_JSON} ..."
        TEMP_JSON=$(jq ".\"registry-mirrors\" += [ \"${DOCKER_MIRROR_URL}\" ]" "${DAEMON_JSON}")
        echo "${TEMP_JSON}" > "${DAEMON_JSON}"
    else
        echo "    ## Creating ${DAEMON_JSON} ..."
        mkdir -p "${DAEMON_JSON%/*}"
        echo "{ \"registry-mirrors\": [ \"${DOCKER_MIRROR_URL}\" ] }" > "${DAEMON_JSON}"
    fi
fi

echo '  - Starting rootless Docker daemon ...'
systemctl --user start docker

echo '  - Enabling Docker service and launch the daemon on startup ...'
systemctl --user enable docker
sudo loginctl enable-linger "$(whoami)"
