#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Configure APT and network for speedup (**reboot required**)
#
# The following environment variables may need to be set:
# * VM_HTTP_PROXY
# * VM_HTTPS_PROXY
# * VM_PROXY_CERTIFICATE
# 
# See https://www.compshare.cn/docs/operation/gpu/uaaa
######################################################################
set -euo pipefail
shopt -s inherit_errexit

echo '* Configuring APT lock ...'
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer
sudo systemctl mask apt-daily.timer apt-daily-upgrade.timer
sudo systemctl stop apt-daily.service apt-daily-upgrade.service
sudo systemctl mask apt-daily.service apt-daily-upgrade.service

if [[ -n "${VM_PROXY_CERTIFICATE:-}" ]]; then
    echo '* Adding CA certificate for HTTPS proxy ...'
    echo '  + Installing prerequisites ...'
    while sudo fuser /var/lib/apt/lists/lock 2>/dev/null; do
        echo '    - Waiting for processes releasing /var/lib/apt/lists/lock ...'
        sleep 5
    done
    sudo apt-get update
    count=0
    until sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
        apt-get install -y ca-certificates; do
        echo '    - Failed to install.'
        count=$((count + 1))
        if [[ $count -lt 10 ]]; then
            sleep 5
            echo '    - Trying again ...'
        else
            exit 1
        fi
    done

    echo '  + Updating CA certificate ...'
    echo "${VM_PROXY_CERTIFICATE}" \
        | sudo tee /usr/local/share/ca-certificates/vm_proxy_cert.crt > /dev/null
    sudo update-ca-certificates
fi

if [[ -n "${VM_HTTP_PROXY:-}" || -n "${VM_HTTPS_PROXY:-}" ]]; then
    echo '* Configuring system-wide proxies ...'
    echo '  + Configuring no proxy ...'
    sudo tee -a /etc/environment > /dev/null << EOF
no_proxy="mirrors.ucloud.cn,developer.download.nvidia.com"
NO_PROXY="mirrors.ucloud.cn,developer.download.nvidia.com"
EOF

    if [[ -n "${VM_HTTP_PROXY:-}" ]]; then
        echo '  + Configuring HTTP proxy ...'
        sudo tee -a /etc/environment > /dev/null << EOF
http_proxy="${VM_HTTP_PROXY}"
HTTP_PROXY="${VM_HTTP_PROXY}"
EOF
    fi

    if [[ -n "${VM_HTTPS_PROXY:-}" ]]; then
        echo '  + Configuring HTTPS proxy ...'
        sudo tee -a /etc/environment > /dev/null << EOF
https_proxy="${VM_HTTPS_PROXY}"
HTTPS_PROXY="${VM_HTTPS_PROXY}"
EOF
    fi
fi

echo '* Adding extra DNS ...'
sudo awk -i inplace \
    '/nameservers:/ {start=1}; \
    start && /addresses:/ && !done { \
        print; \
        print "                - 100.90.90.90"; \
        print "                - 100.90.90.100"; \
        done=1; \
        next \
    } 1' \
    /etc/netplan/50-cloud-init.yaml

echo '* Applying network configuration ...'
sudo netplan apply
