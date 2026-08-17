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

echo '* Importing utility functions ...'
source "$(dirname "$0")/utils.sh"

os="$(. /etc/os-release && echo "${NAME}${VERSION_ID}" | tr -d '.' | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"
cuda_repo="https://developer.download.nvidia.com/compute/cuda/repos"
cuda_keyring="cuda-keyring_1.1-1_all.deb"
cuda_keyring_url="${cuda_repo}/${os}/${arch}/${cuda_keyring}"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry gcc "linux-headers-$(uname -r)"

echo '* Installing cuda-keyring ...'
run_cmd_retry curl -fsSL "${cuda_keyring_url}" -o "${cuda_keyring}"
sudo dpkg -i "${cuda_keyring}"
rm -f "${cuda_keyring}"
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
    # GPU architecture from and above Turing (including 2080 and T4)
    # should use open kernel modules.
    apt_install_retry libnvidia-compute nvidia-dkms-open
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
