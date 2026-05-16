#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Configure APT and network for speedup (**reboot required**)
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
