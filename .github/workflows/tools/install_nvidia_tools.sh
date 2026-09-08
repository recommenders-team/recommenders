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

script_dir="$(dirname "$0")"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
# Check if the machine is CUDA capable.
#--------------------------------------------------------------------
echo '* Checking if there are any NVIDIA GPUs ...'
sudo update-pciids
if ! lspci | grep -i nvidia > /dev/null; then
    exit 0
fi


#--------------------------------------------------------------------
# Install prerequisites
#--------------------------------------------------------------------
echo '* Importing utility functions ...'
source "${utils_sh}"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry ca-certificates curl gnupg


#--------------------------------------------------------------------
# Install CUDA driver if it is not installed.
#--------------------------------------------------------------------
if ! nvidia-smi 2>/dev/null; then
    os="$(. /etc/os-release \
        && echo "${NAME}${VERSION_ID}" \
            | tr -d '.' | tr '[:upper:]' '[:lower:]')"
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
    if lspci | grep -i nvidia | grep -Ei 'p40|v100s'; then
        # P40 can only install drivers of version up to 580
        echo '  + Locking to version 580 ...'
        apt_install_retry nvidia-driver-pinning-580

        echo '  + Installing compute-only drivers ...'
        apt_install_retry libnvidia-compute-580 nvidia-dkms-580
    else
        # GPU architecture from and above Turing (including 2080 and
        # T4) should use open kernel modules.
        apt_install_retry libnvidia-compute nvidia-dkms-open
    fi
fi


#--------------------------------------------------------------------
# Install NVIDIA container toolkit if it is not installed.
#--------------------------------------------------------------------
if nvidia-ctk --version 2>/dev/null; then
    echo '* Installing NVIDIA container toolkit ...'
    apt_install_retry \
        nvidia-container-toolkit \
        nvidia-container-toolkit-base \
        libnvidia-container-tools \
        libnvidia-container1

    echo '* Configuring the container runtime ...'
    if docker info 2>/dev/null | grep -i rootless > /dev/null; then
        nvidia-ctk runtime configure --runtime=docker \
            --config="${HOME}/.config/docker/daemon.json"
        systemctl --user restart docker
        sudo nvidia-ctk config --in-place \
            --set nvidia-container-cli.no-cgroups 
    else
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
    fi
fi
