#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Get the test groups from test group configuration file.
# 
# Params:
#   * Path to test group configuration file relative to the repo root
#   * Type of test - pr_gate or nightly
#   * Test compute - cpu, gpu or spark.  It is only used for nightly
#     not pr_gate
#
# This script is used in GitHub Actions workflows and uses
# GITHUB_OUTPUT to output variables for subsequent jobs.
######################################################################
set -euo pipefail
shopt -s inherit_errexit

config_file="${1:-}"
test_type="${2:-}"
compute="${3:-}"

[[ -z ${config_file} \
  || -z ${test_type} \
  || -z ${compute} ]] && exit 1

if [[ ${test_type} == 'nightly' ]]; then
    test_groups_str=$(yq -o json -I 0 "
        [ .${test_type}
        | keys
        | .[]
        | select(contains(\"${compute}\"))
        ]" \
        "${config_file}")
else
    test_groups_str=$(yq -o json -I 0 \
        ".${test_type} | keys" "${config_file}")
fi
echo "Test Groups: ${test_groups_str}"
echo "groups=${test_groups_str}" >> ${GITHUB_OUTPUT}
