#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Get the test groups from test group configuration file.
#
# NOTE:
# * This script is used in GitHub Actions workflows and uses
#   GITHUB_OUTPUT to output variables for subsequent jobs.
# * yq is required to run this script, and it is preinstalled on the
#   GitHub Actions runner image ubuntu-24.04.  See
#   https://github.com/actions/runner-images/blob/9b8c9709431a8d2b295fd46cf96f283d671f20d9/images/ubuntu/Ubuntu2404-Readme.md?plain=1#L100
#   
#
# Params:
#   * Path to test group configuration file relative to the repo root
#   * Type of test - pr_gate or nightly
#   * Test compute - cpu, gpu or spark.  It is only used for nightly
#     not pr_gate
######################################################################
set -euo pipefail
shopt -s inherit_errexit

test_groups_yml="${1:-}"
test_type="${2:-}"
compute="${3:-}"

[[ -z ${test_groups_yml} \
  || -z ${test_type} \
  || -z ${compute} ]] && exit 1


#--------------------------------------------------------------------
# Get the test groups according to test type and compute
#--------------------------------------------------------------------
if [[ ${test_type} == 'nightly' ]]; then
    test_groups_str=$(yq -o json -I 0 "
        [ .${test_type}
        | keys
        | .[]
        | select(contains(\"${compute}\"))
        ]" \
        "${test_groups_yml}")
else
    test_groups_str=$(yq -o json -I 0 \
        ".${test_type} | keys" "${test_groups_yml}")
fi
echo "Test Groups: ${test_groups_str}"
echo "groups=${test_groups_str}" >> ${GITHUB_OUTPUT}
