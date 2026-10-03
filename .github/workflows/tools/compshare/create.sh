#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Create a CompShare VM.
#
# NOTE:
# * The script must set the environment variable SSH_DEST into
#   $GITHUB_ENV for subsequent steps.
# * It is assumed that there is a configuration file called
#
#     config.yml
#
#   in a directory named '${CLOUD_SERVICE@L}' under the
#   script directory.  In config.yml, the following key may need to be
#   set:
#   + secret_key_name
#     - the name of the secret of private key for the cloud service
#       indicated by the environment variable CLOUD_SERVICE.
#
# Params:
# * VM name
# * Test type
# * Test group
# * requirements in JSON.  It is not intended to be used in the
#   testing workflow.
#   + {"GPUType":"!2080,P40","Memory":10240,"GraphicsMemory":10240}
#     - It means the GPUType should not be 2080 and P40,
#       GPU memory should >= 10240MB
#       and CPU 10240MB.
#   + {"GPUType":"2080,P40"}
#     - It means the GPUType should be 2080 or P40.
#   + {"GPUType":["2080","P40"],"ChargeType":"Spot"}
#     - It means the GPUType should be 2080 or P40,
#       ChargeType should be Spot.
#
# The following environment variables must be set:
# * CLOUD_SERVICE
#   + It should be the name of the directory containing the
#     configuration files for the cloud service, such as alicloud and
#     compshare.
# * CLOUD_SERVICE_SECRET
#   + It contains the private key for CompShare APIs and is used as
#     COMPSHARE_PRIVATE_KEY in the script.
# * CLOUD_SERVICE_ENVS
#   + It contains the following keys in JSON:
#     - COMPSHARE_PUBLIC_KEY (required)
#
# The following environment variables may need to be set:
# * CLOUD_SERVICE_INPUT_VARS
#   + It contains the possible values of the input variables for
#     creating the VM, in the JSON format like the following:
#
#     {
#         "GpuType": [
#             "3080Ti",
#             "3090",
#             "4090",
#             "5090",
#             "4090_48G"
#         ],
#         "Zone": [
#             "cn-wlcb-01",
#             "cn-sh2-02"
#         ],
#         "ChargeType": [
#             "Spot",
#             "Postpay"
#         ]
#     }
#
#   + Possible keys are
#     - GpuType
#     - Zone
#     - ChargeType
#   + Possible values of Zone can be found at
#     https://www.compshare.cn/docs/gpus/instance/describecompsharesupportzone
#   + Possible values of GpuType can be found at
#     https://www.compshare.cn/docs/gpus/instance/createcompshareinstance#gpu-类型列表
#   + Possible values of ChargeType can be found at
#     https://www.compshare.cn/docs/gpus/instance/createcompshareinstance#计费参数
######################################################################
set -euo pipefail
shopt -s inherit_errexit

script_path="$(realpath -- "${BASH_SOURCE[0]}")"
script_dir="$(dirname -- "${script_path}")"
unique_name="${1:-}"
test_type="${2:-}"
test_group="{3:-}"
requirements="${4:-}"

[[ -z ${unique_name} \
  || -z ${test_type} \
  || -z ${test_group} \
  || -z ${CLOUD_SERVICE:-} ]] \
    && { echo 'Parameter error!' >&2; exit 1; }

cloud_service="${CLOUD_SERVICE}"
cloud_service="${cloud_service@L}"
config_dir="${script_dir}/${cloud_service}"
config_yml="${config_dir}/config.yml"
utils_sh="${script_dir}/utils.sh"


#--------------------------------------------------------------------
echo 'Importing utility functions ...'
source "${utils_sh}"

echo 'Exporting environment variables ...'
secret_key_name="$(yq '.secret_key_name' < "${config_yml}")"
export "${secret_key_name}"="${CLOUD_SERVICE_SECRET}"
eval "$(get_env_exports "${CLOUD_SERVICE_ENVS:-}")"


#--------------------------------------------------------------------
# Use CompShare APIs to create a VM.
#--------------------------------------------------------------------
if [[ -z ${requirements} ]]; then
    # * All VMs from CompShare have GPUs.
    # * Unit tests require less than half hour
    # * Nightly tests take more than an hour and more GPU memory.
    # * VMs with Spot ChargeType are cheaper but there is a risk of
    #   being deleted after 1 hour.
    if [[ ${test_type} == *nightly* ]]; then
        requirements='{
            "GpgType": "!2080,P40",
            "Memory": 32,
            "GraphicsMemory": 12,
            "ChargeType": ["Postpay"]
        }'
    else
        requirements='{
            "GpuType": "!P40",
            "Memory": 32,
            "GraphicsMemory": 8,
            "ChargeType": ["Spot","Postpay"]
        }'
    fi
fi

cloud_service_input_vars="${CLOUD_SERVICE_INPUT_VARS:-}"
if [[ ${test_group} == *gpu* ]]; then
    input_vars="$(jq '.gpu // empty' \
        <<< "${cloud_service_input_vars}")"
else
    input_vars="$(jq '.cpu // empty' \
        <<< "${cloud_service_input_vars}")"
fi
input_vars="${input_vars:-$cloud_service_input_vars}"

allocate_vm "${unique_name}" "${requirements}" "${input_vars}"

echo "Getting info of the VM ..."
vm_info="$(get_vm_info "${unique_name}")"
[[ -z ${vm_info} ]] && exit 1

echo 'Exporting VM info for subsequent steps ...'
ssh_dest="$(jq -r '.SshLoginCommand
    | split(" +"; null)
    | .[]
    | select(contains("@"))' <<< "${vm_info}")"
echo "SSH_DEST=${ssh_dest}" >> "$GITHUB_ENV"

echo 'Setting stop scheduler ...'
if [[ ${test_type} == *nightly* ]]; then
    stop_time="$(date --date='3 hours' '+%s')"
else
    stop_time="$(date --date='1 hours' '+%s')"
fi
stop_time="{\"SchedulerStopTime\": ${stop_time}}"
stop_scheduler="$(update_json "${vm_info}" "${stop_time}")"
api_call_retry update_stop_scheduler "${stop_scheduler}" > /dev/null

unset "${secret_key_name}"

wait_for_vm_to_be_available "${ssh_dest}"
encoded_password="$(jq '.Password' <<< "${vm_info}")"
setup_ssh_key "${ssh_dest}" "${encoded_password}"
