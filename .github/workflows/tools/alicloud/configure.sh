#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Configuration specific to AliCloud
######################################################################
set -euo pipefail
shopt -s inherit_errexit

echo '* Importing utility functions ...'
source "$(dirname "$0")/../utils.sh"

echo '* Installing prerequisites ...'
wait_for_apt_lock
sudo apt-get update
apt_install_retry git-all
