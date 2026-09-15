#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Install Docker or not if already installed.
#
# It assumes that there is a configuration file called
#
#     config.yml
#
# in a directory named '${CLOUD_SERVICE@L}' under the
# script directory.  In config.yml, the following key may need to be
# set:
# * docker_download_mirror
#   + the mirror to be used for downloading Docker installation
#     packages.
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * VM_DOCKER_MIRROR_URL
#   + semicolon separated URLs
#
# See
# * https://docs.docker.com/engine/install/ubuntu/
# * https://docs.docker.com/engine/security/rootless/
# * Docker CE mirror at AliCloud
#   + https://developer.aliyun.com/mirror/docker-ce
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"

cloud_service="${CLOUD_SERVICE:-}"
cloud_service="${cloud_service@L}"
config_yml="${script_dir}/${cloud_service}/config.yml"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
# Install prerequisites
#--------------------------------------------------------------------
echo '* Importing utility functions ...'
source "${utils_sh}"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry ca-certificates curl gnupg jq yq

if [[ $(whoami) == 'root' ]]; then
    rootless='false'
else
    rootless='true'
fi


#--------------------------------------------------------------------
# Install Docker if it is not installed.
#--------------------------------------------------------------------
if ! docker --version 2>/dev/null; then
    arch="$(dpkg --print-architecture)"
    codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"

    if [[ -f ${config_yml} ]]; then
        apt_url="$(yq '.docker_download_mirror // ""' "${config_yml}")"
    fi
    apt_url="${apt_url:-https://download.docker.com/linux/ubuntu}"
    apt_list="/etc/apt/sources.list.d/docker.list"
    keyring_dir="/etc/apt/keyrings"
    gpg_path="${keyring_dir}/docker.asc"
    gpg_url="${apt_url}/gpg"
    apt_entry="deb [arch=${arch} signed-by=${gpg_path}] ${apt_url} ${codename} stable"

    echo '* Adding Docker official GPG key ...'
    sudo install -m 0755 -d "${keyring_dir}"

    run_cmd_retry 10 sudo curl -fsSL "${gpg_url}" -o "${gpg_path}"
    sudo chmod a+r "${gpg_path}"

    echo '* Setting APT repo source for Docker ...'
    sudo mkdir -p "$(dirname "${apt_list}")"
    echo "${apt_entry}" | sudo tee "${apt_list}" > /dev/null
    sudo apt-get update

    echo '* Installing the latest Docker community edition ...'
    apt_install_retry docker-ce

    if [[ ${rootless} == 'true' ]]; then
        echo '* Configuring Docker daemon in rootless mode ...'
        echo '  - Installing prerequisites ...'
        apt_install_retry uidmap docker-ce-rootless-extras

        echo '  - Disabling system-wide Docker daemon ...'
        sudo systemctl disable --now docker.service docker.socket
        sudo rm /var/run/docker.sock

        echo '  - Installing rootless Docker daemon ...'
        dockerd-rootless-setuptool.sh install
    fi
fi


#--------------------------------------------------------------------
# Check whether Docker is in rootless mode instead of using the
# setting in config.yml in case Docker is preinstalled.
#--------------------------------------------------------------------
if docker info 2>/dev/null | grep -i rootless > /dev/null; then
    rootless=true
else
    rootless=false
fi


#--------------------------------------------------------------------
# Configure Docker mirrors if VM_DOCKER_MIRROR_URL is provided.
#--------------------------------------------------------------------
if [[ -n ${VM_DOCKER_MIRROR_URL:-} ]]; then
    echo '* Setting Docker mirror URL ...'
    if [[ ${rootless} == 'true' ]]; then
        daemon_json="${HOME}/.config/docker/daemon.json"
    else
        daemon_json="/etc/docker/daemon.json"
    fi
    shopt -s extglob
    mirrors="${VM_DOCKER_MIRROR_URL//*( );*( )/\",\"}"
    shopt -u extglob
    updates="{ \"registry-mirrors\": [ \"${mirrors}\" ] }"
    if [[ -f ${daemon_json} ]]; then
        echo "  ## Updating ${daemon_json} ..."
        res="$(update_json <(cat "${daemon_json}") "${updates}")"
        echo "${res}" > "${daemon_json}"
    else
        echo "  ## Creating ${daemon_json} ..."
        mkdir -p "$(dirname "${daemon_json}")"
        jq '.' > "${daemon_json}" <<< "${updates}"
    fi
fi


#--------------------------------------------------------------------
# Start the Docker service.
#--------------------------------------------------------------------
if [[ ${rootless} == 'true' ]]; then
    echo '* Starting rootless Docker daemon ...'
    systemctl --user restart docker

    echo '* Enabling Docker service and launch the daemon on startup ...'
    systemctl --user enable docker
    sudo loginctl enable-linger "$(whoami)"
else
    echo '* Starting Docker daemon ...'
    sudo systemctl restart docker
fi
