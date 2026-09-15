#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Configure APT and network for speedup.
#
# It assumes that there is a configuration file called
#
#     config.yml
#
# in a directory named '${CLOUD_SERVICE@L}' under the
# script directory.  In config.yml, the following key may need to be
# set:
# * apt_mirror
#   + the APT mirror to be used for downloading system packages.
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * VM_HTTP_PROXY
# * VM_HTTPS_PROXY
# * VM_PROXY_CERTIFICATE
# 
# See https://www.compshare.cn/docs/operation/gpu/uaaa
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_dir="$(dirname "$0")"

cloud_service="${CLOUD_SERVICE:-}"
cloud_service="${cloud_service@L}"
config_yml="${script_dir}/${cloud_service}/config.yml"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
# Disable auto updates
#--------------------------------------------------------------------
echo '* Importing utility functions ...'
source "${utils_sh}"

echo '* Configuring APT lock ...'
sudo systemctl stop apt-daily.timer apt-daily-upgrade.timer
sudo systemctl mask apt-daily.timer apt-daily-upgrade.timer
sudo systemctl stop apt-daily.service apt-daily-upgrade.service
sudo systemctl mask apt-daily.service apt-daily-upgrade.service


#--------------------------------------------------------------------
# Install prerequisites
#--------------------------------------------------------------------
echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry ca-certificates curl git-all jq yq


#--------------------------------------------------------------------
# Configure HTTP/HTTPS proxies if any
#--------------------------------------------------------------------
if [[ -n ${VM_PROXY_CERTIFICATE:-} ]]; then
    echo '* Adding CA certificate for HTTPS proxy ...'
    echo "${VM_PROXY_CERTIFICATE}" \
        | sudo tee /usr/local/share/ca-certificates/vm_proxy_cert.crt > /dev/null
    sudo update-ca-certificates
fi

if [[ -n ${VM_HTTP_PROXY:-} || -n ${VM_HTTPS_PROXY:-} ]]; then
    echo '* Configuring system-wide proxies ...'
    echo '  + Configuring no proxy ...'
    if [[ -f ${config_yml} ]]; then
        apt_mirror="$(yq '.apt_mirror // ""' "${config_yml}")"
    fi
    apt_mirror="${apt_mirror:+$apt_mirror,}"
    sudo tee -a /etc/environment > /dev/null << EOF
no_proxy="${apt_mirror}developer.download.nvidia.com"
NO_PROXY="${apt_mirror}developer.download.nvidia.com"
EOF

    if [[ -n ${VM_HTTP_PROXY:-} ]]; then
        echo '  + Configuring HTTP proxy ...'
        sudo tee -a /etc/environment > /dev/null << EOF
http_proxy="${VM_HTTP_PROXY}"
HTTP_PROXY="${VM_HTTP_PROXY}"
EOF
    fi

    if [[ -n ${VM_HTTPS_PROXY:-} ]]; then
        echo '  + Configuring HTTPS proxy ...'
        sudo tee -a /etc/environment > /dev/null << EOF
https_proxy="${VM_HTTPS_PROXY}"
HTTPS_PROXY="${VM_HTTPS_PROXY}"
EOF
    fi
fi


#--------------------------------------------------------------------
# Network configs for CompShare VMs
#--------------------------------------------------------------------
if [[ ${cloud_service} == 'compshare' ]]; then
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
    sudo resolvectl flush-caches
fi
