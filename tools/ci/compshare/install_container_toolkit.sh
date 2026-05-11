#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Install NVIDIA container toolkit (**reboot required**)
#
# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
######################################################################
set -euo pipefail
shopt -s inherit_errexit

KEYRING_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
KEYRING_PATH="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
NCT_LIST_FILE="nvidia-container-toolkit.list"
NCT_URL="https://nvidia.github.io/libnvidia-container/stable/deb/${NCT_LIST_FILE}"
APT_LIST_DIR='/etc/apt/sources.list.d'

echo '* Getting NVIDIA container toolkit GPG key ...'
curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors "${KEYRING_URL}" -o "${KEYRING_URL##*/}"
sudo gpg --dearmor --yes -o "${KEYRING_PATH}" "${KEYRING_URL##*/}"
rm -rf "${KEYRING_URL##*/}"

echo '* Setting APT repo source for NVIDIA container toolkit ...'
sudo mkdir -p "${APT_LIST_DIR}"
curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors "${NCT_URL}" -o "${NCT_LIST_FILE}"
sed -i "s#deb https://#deb [signed-by=${KEYRING_PATH}] https://#g" "${NCT_LIST_FILE}"
sudo mv "${NCT_LIST_FILE}" "${APT_LIST_DIR}"
sudo apt-get update

echo '* Installing NVIDIA container toolkit ...'
sudo dpkg --configure -a
count=0
until sudo apt-get install -y \
        nvidia-container-toolkit \
        nvidia-container-toolkit-base \
        libnvidia-container-tools \
        libnvidia-container1; do
    echo '  + Failed to install.'
    count=$((count + 1))
    if [[ $count -lt 5 ]]; then
        sleep 5
        echo '  + Trying again ...'
    else
        exit 1
    fi
done

echo '* Configuring the container runtime ...'
nvidia-ctk runtime configure --runtime=docker --config="${HOME}/.config/docker/daemon.json"
sudo systemctl restart docker
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
