#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Install Docker in rootless mode
#
# The following environment variables may need to be set:
# * VM_DOCKER_MIRROR_URL
#
# See
# * https://docs.docker.com/engine/install/ubuntu/
# * https://docs.docker.com/engine/security/rootless/
######################################################################
set -euo pipefail
shopt -s inherit_errexit

SCRIPT_DIR="$(dirname "$0")"

# Utility functions
SCRIPT_UTILS="${SCRIPT_DIR}/utils.sh"

echo '* Importing utility functions ...'
source "${SCRIPT_UTILS}"

ARCH="$(dpkg --print-architecture)"
CODENAME="$(. /etc/os-release && echo "$VERSION_CODENAME")"
APT_URL="https://download.docker.com/linux/ubuntu"
APT_LIST="/etc/apt/sources.list.d/docker.list"
KEYRING_DIR="/etc/apt/keyrings"
GPG_PATH="${KEYRING_DIR}/docker.asc"
GPG_URL="${APT_URL}/gpg"
APT_ENTRY="deb [arch=${ARCH} signed-by=${GPG_PATH}] ${APT_URL} ${CODENAME} stable"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry ca-certificates curl jq

echo '* Adding Docker official GPG key ...'
sudo install -m 0755 -d "${KEYRING_DIR}"
sudo curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors "${GPG_URL}" -o "${GPG_PATH}"
sudo chmod a+r "${GPG_PATH}"

echo '* Setting APT repo source for Docker ...'
sudo mkdir -p "$(dirname "${APT_LIST}")"
echo "${APT_ENTRY}" | sudo tee "${APT_LIST}" > /dev/null
sudo apt-get update

echo '* Installing the latest Docker community edition ...'
apt_install_retry docker-ce

echo '* Configuring Docker daemon in rootless mode ...'
echo '  - Installing prerequisites ...'
apt_install_retry uidmap docker-ce-rootless-extras

echo '  - Disabling system-wide Docker daemon ...'
sudo systemctl disable --now docker.service docker.socket
sudo rm /var/run/docker.sock

echo '  - Installing rootless Docker daemon ...'
dockerd-rootless-setuptool.sh install

if [[ -n "${VM_DOCKER_MIRROR_URL:-}" ]]; then
    echo '* Setting Docker mirror URL ...'
    daemon_json="${HOME}/.config/docker/daemon.json"
    updates="{ \"registry-mirrors\": [ \"${VM_DOCKER_MIRROR_URL}\" ] }"
    if [[ -f "${daemon_json}" ]]; then
        echo "  ## Updating ${daemon_json} ..."
        res="$(update_json <(cat "${daemon_json}") "${updates}")"
        echo "${res}" > "${daemon_json}"
    else
        echo "  ## Creating ${daemon_json} ..."
        mkdir -p "$(dirname "${daemon_json}")"
        echo "${updates}" | jq '.' > "${daemon_json}"
    fi
fi

echo '* Starting rootless Docker daemon ...'
systemctl --user start docker

echo '* Enabling Docker service and launch the daemon on startup ...'
systemctl --user enable docker
sudo loginctl enable-linger "$(whoami)"
