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

OS="$(. /etc/os-release && echo "${NAME}${VERSION_ID}" | tr -d '.' | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
CUDA_REPO="https://developer.download.nvidia.com/compute/cuda/repos"
CUDA_KEYRING="cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_URL="${CUDA_REPO}/${OS}/${ARCH}/${CUDA_KEYRING}"

echo '* Installing prerequisites ...'
while sudo fuser /var/lib/apt/lists/lock 2>/dev/null; do
    echo '    - Waiting for processes releasing /var/lib/apt/lists/lock ...'
    sleep 5
done
sudo apt-get update
count=0
until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
    apt-get install -y gcc "linux-headers-$(uname -r)"; do
    echo '  + Failed to install.'
    count=$((count + 1))
    if [[ $count -lt 5 ]]; then
        sleep 5
        echo '  + Trying again ...'
    else
        exit 1
    fi
done

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
    count=0
    until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
        apt-get install -y nvidia-driver-pinning-580; do
        echo '  + Failed to install.'
        count=$((count + 1))
        if [[ $count -lt 5 ]]; then
            sleep 5
            echo '  + Trying again ...'
        else
            exit 1
        fi
    done

    echo '  + Installing compute-only drivers ...'
    count=0
    until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
        apt-get install -y libnvidia-compute-580 nvidia-dkms-580; do
        echo '  + Failed to install.'
        count=$((count + 1))
        if [[ $count -lt 5 ]]; then
            sleep 5
            echo '  + Trying again ...'
        else
            exit 1
        fi
    done
else
    count=0
    until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
        apt-get install -y libnvidia-compute nvidia-dkms; do
        echo '  + Failed to install.'
        count=$((count + 1))
        if [[ $count -lt 5 ]]; then
            sleep 5
            echo '  + Trying again ...'
        else
            exit 1
        fi
    done
fi

echo '* Installing NVIDIA container toolkit ...'
count=0
until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
    apt-get install -y \
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
