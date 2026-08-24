#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Configuration specific to CompShare
# 
# See https://www.compshare.cn/docs/operation/gpu/uaaa
######################################################################
set -euo pipefail
shopt -s inherit_errexit

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
