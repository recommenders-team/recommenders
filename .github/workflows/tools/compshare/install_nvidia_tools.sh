#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Install NVIDIA CUDA driver and container toolkit (**reboot required**)
#
# See
# * https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
# * https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html
######################################################################
set -euo pipefail
shopt -s inherit_errexit

SCRIPT_DIR="$(dirname "$0")"

# Utility functions
SCRIPT_UTILS="${SCRIPT_DIR}/utils.sh"

echo '* Importing utility functions ...'
source "${SCRIPT_UTILS}"

OS="$(. /etc/os-release && echo "${NAME}${VERSION_ID}" | tr -d '.' | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos"
CUDA_KEYRING="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="${CUDA_REPO}/${OS}/${ARCH}/${CUDA_KEYRING}"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry gcc "linux-headers-$(uname -r)"

echo '* Installing cuda-keyring ...'
curl -fsSL --retry 5 --retry-delay 10 --retry-all-errors "${CUDA_KEYRING_URL}" -o "${CUDA_KEYRING}"
sudo dpkg -i "${CUDA_KEYRING}"
rm -f "${CUDA_KEYRING}"
sudo apt-get update

echo '* Installing CUDA driver ...'
sudo update-pciids

if lspci | grep -i nvidia | grep -Ei 'p40|v100s'; then
    # P40 can only install drivers of version up to 580
    echo '  + Locking to version 580 ...'
    apt_install_retry nvidia-driver-pinning-580

    echo '  + Installing compute-only drivers ...'
    apt_install_retry libnvidia-compute-580 nvidia-dkms-580
else
    apt_install_retry libnvidia-compute nvidia-dkms
fi

echo '* Installing NVIDIA container toolkit ...'
apt_install_retry \
    nvidia-container-toolkit \
    nvidia-container-toolkit-base \
    libnvidia-container-tools \
    libnvidia-container1

echo '* Configuring the container runtime ...'
nvidia-ctk runtime configure --runtime=docker --config="${HOME}/.config/docker/daemon.json"
sudo systemctl restart docker
sudo nvidia-ctk config --set nvidia-container-cli.no-cgroups --in-place
