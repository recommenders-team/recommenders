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
NCT_FILE="nvidia-container-toolkit.list"

echo '* Getting NVIDIA container toolkit GPG key ...'
curl -fsSL "${KEYRING_URL}" | sudo gpg --dearmor --yes -o "${KEYRING_PATH}"

echo '* Setting APT repo source for NVIDIA container toolkit ...'
curl -s -L "${NCT_URL}" \
  | sed "s#deb https://#deb [signed-by=${KEYRING_PATH}] https://#g" \
  | sudo tee "/etc/apt/sources.list.d/${NCT_FILE}"
sudo apt update

echo '* Installing NVIDIA container toolkit ...'
sudo apt install -y \
    nvidia-container-toolkit \
    nvidia-container-toolkit-base \
    libnvidia-container-tools \
    libnvidia-container1

echo '* Configuring the container runtime ...'
nvidia-ctk runtime configure --runtime=docker --config="${HOME}/.config/docker/daemon.json"
sudo systemctl restart docker
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
