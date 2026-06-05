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

echo '* Importing utility functions ...'
source "$(dirname "$0")/utils.sh"

arch="$(dpkg --print-architecture)"
codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"
apt_url="https://download.docker.com/linux/ubuntu"
apt_list="/etc/apt/sources.list.d/docker.list"
keyring_dir="/etc/apt/keyrings"
gpg_path="${keyring_dir}/docker.asc"
gpg_url="${apt_url}/gpg"
apt_entry="deb [arch=${arch} signed-by=${gpg_path}] ${apt_url} ${codename} stable"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry ca-certificates curl jq

echo '* Adding Docker official GPG key ...'
sudo install -m 0755 -d "${keyring_dir}"

run_cmd_retry sudo curl -fsSL "${gpg_url}" -o "${gpg_path}"
sudo chmod a+r "${gpg_path}"

echo '* Setting APT repo source for Docker ...'
sudo mkdir -p "$(dirname "${apt_list}")"
echo "${apt_entry}" | sudo tee "${apt_list}" > /dev/null
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
        jq '.' > "${daemon_json}" <<< "${updates}"
    fi
fi

echo '* Starting rootless Docker daemon ...'
systemctl --user start docker

echo '* Enabling Docker service and launch the daemon on startup ...'
systemctl --user enable docker
sudo loginctl enable-linger "$(whoami)"
