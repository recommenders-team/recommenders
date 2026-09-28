#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Utils for CompShare APIs
#
# The following environment variables must be set when using these
# functions:
# * COMPSHARE_PRIVATE_KEY
# * COMPSHARE_PUBLIC_KEY
######################################################################

# Source common utils
source "$(dirname "$0")/../utils.sh"

# Constants
COMPSHARE_ZONE_CHINA_NORTH_2A='cn-wlcb-01'
COMPSHARE_IMAGE_UBUNTU2404='compshareImage-12rjyhwynazd'


#---------------------------------------------------------------------
# Utils used by other CompShare API wrappers and utils
#---------------------------------------------------------------------
check_vm_requirement() {
    # Check if the VM specification match the requirements.
    #
    # Params:
    # * VM specification in JSON
    # * requirements in JSON, for example
    #   + {"GPUType":"!2080,P40","Memory":{"GPU":10,"CPU":9}}
    #     - It means the GPUType should not be 2080 and P40,
    #       GPU memory should be greater than or equal to 10GB
    #       and CPU 9GB.
    #   + {"GPUType":"2080,P40"}
    #     - It means the GPUType should be 2080 or P40.
    local spec="${1:-}"
    local requirements="${2:-}"

    local match
    match=$(jq -s '
        def equalstr($a; $b):
            if ($a | startswith(" ")) then
                equalstr(($a | ltrimstr(" ")); $b)
            elif ($a | endswith(" ")) then
                equalstr(($a | rtrimstr(" ")); $b)
            else
                $a == $b
            end;
        def compareitem($req; $spec; $i):
            ($req | getpath($i)) as $a
            | ($spec | getpath($i)) as $b
            | ($a | type) as $ta
            | if $ta == "string" then
                $a | if startswith("!") then
                    $a | ltrimstr("!") | split(",")
                    | reduce .[] as $i (true; . and (equalstr($i; $b) | not))
                    | if . then . else debug("Demand (\($b)) should not be any one of (\($i) - \($a))") end
                else
                    $a | split(",")
                    | reduce .[] as $i (false; . or equalstr($i; $b))
                    | if . then . else debug("Demand (\($b)) must be one of (\($i) - \($a))") end
                end
            elif $ta == "number" then
                $a <= $b | if . then . else debug("Demand (\($b)) should be greater than or equal to (\($i) - \($a))") end
            else
                true
            end;
        .[0] as $req
        | .[1] as $spec
        | .[0] | [path(..)]
        | reduce .[] as $i (true; . and compareitem($req; $spec; $i))' \
        <(echo "${requirements}") <(echo "${spec}"))

    echo "${match}"
}

gen_action_digest() {
    # Generate the digest for the action requrest parameters
    # See https://docs.ucloud.cn/api/summary/signature
    # 
    # Params:
    # * API action specification in JSON
    # * (Optional) file containing the base64-encoded login password
    local action_spec="${1:-}"
    local encoded_password_file="${2:-}"
    [[ -z ${action_spec} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    # Store the spec into a file to hide the password from being
    # visible
    local action_spec_file
    action_spec_file="$(mktemp)"
    trap "rm -f '${action_spec_file}'; trap - EXIT RETURN" EXIT RETURN
    echo "${action_spec}" > "${action_spec_file}"
    if [[ -n ${encoded_password_file} ]]; then
        echo "${action_spec}" \
            | jq --rawfile encoded_password \
                "${encoded_password_file}" \
                '.Password = $encoded_password' \
                > "${action_spec_file}"
    fi

    local reset_x=false
    [[ "$-" == *x* ]] && reset_x=true
    set +x

    # COMPSHARE_PRIVATE_KEY are set as an environment variable,
    # not directly in the script
    local digest
    digest="$(\
        jq -r '
            to_entries
            | sort
            | map("\(.key)\(.value)")
            | join("")' "${action_spec_file}" \
        | tr -d '\n' \
        | cat - <(echo "${COMPSHARE_PRIVATE_KEY}") \
        | tr -d '\n' \
        | sha1sum \
        | head -c 40)"

    [[ "${reset_x}" == true ]] && set -x

    echo "${digest}"
}

gen_request_url() {
    # Generate the API request URL using the action specification and
    # the parameter digest
    #
    # Params:
    # * API action specification in JSON
    # * (Optional) file containing the base64-encoded login password
    local action_spec="${1:-}"
    local encoded_password_file="${2:-}"

    [[ -z ${action_spec} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    # COMPSHARE_PUBLIC_KEY is an environment variable.
    action_spec="$(jq ".PublicKey = \"${COMPSHARE_PUBLIC_KEY}\"" \
        <<< "${action_spec}")"

    local digest
    digest="$(gen_action_digest "${action_spec}" "${encoded_password_file}")"
    local params
    params="$(jq -r '
        to_entries
        | map("\(.key)=\(.value)")
        | join("&")' <<< "${action_spec}")"
    echo "https://api.compshare.cn/?${params}&Signature=${digest}"
}

get_compute_spec() {
    # Return the specification for all available CompShare computes.
    local compute_spec
    compute_spec="$(cat << 'EOF'
        [
            {
                "GPUType": "P40",
                "Memory": {
                    "CPU": 64,
                    "GPU": 24
                },
                "CPU": 8,
                "Price": 0.38,
                "ChargeType": [
                    "Postpay"
                ]
            },
            {
                "GPUType": "2080",
                "Memory": {
                    "CPU": 40,
                    "GPU": 8
                },
                "CPU": 8,
                "Price": 0.39,
                "ChargeType": [
                    "Postpay"
                ]
            },
            {
                "GPUType": "3080Ti",
                "Memory": {
                    "CPU": 32,
                    "GPU": 12
                },
                "CPU": 12,
                "Price": 0.7,
                "ChargeType": [
                    "Postpay",
                    "Spot"
                ]
            },
            {
                "GPUType": "3090",
                "Memory": {
                    "CPU": 64,
                    "GPU": 24
                },
                "CPU": 16,
                "Price": 1.13,
                "ChargeType": [
                    "Postpay",
                    "Spot"
                ]
            },
            {
                "GPUType": "4090",
                "Memory": {
                    "CPU": 64,
                    "GPU": 24
                },
                "CPU": 16,
                "Price": 2.05,
                "ChargeType": [
                    "Postpay",
                    "Spot"
                ]
            },
            {
                "GPUType": "5090",
                "Memory": {
                    "CPU": 96,
                    "GPU": 32
                },
                "CPU": 16,
                "Price": 3.15,
                "ChargeType": [
                    "Postpay",
                    "Spot"
                ]
            },
            {
                "GPUType": "4090_48G",
                "Memory": {
                    "CPU": 96,
                    "GPU": 48
                },
                "CPU": 16,
                "Price": 3.13,
                "ChargeType": [
                    "Postpay"
                ]
            },
            {
                "GPUType": "A800",
                "Memory": {
                    "CPU": 240,
                    "GPU": 80
                },
                "CPU": 16,
                "Price": 6.99,
                "ChargeType": [
                    "Postpay",
                    "Spot"
                ]
            },
            {
                "GPUType": "H20",
                "Memory": {
                    "CPU": 240,
                    "GPU": 96
                },
                "CPU": 16,
                "Price": 7.12,
                "ChargeType": [
                    "Postpay"
                ]
            },
            {
                "GPUType": "A100",
                "Memory": {
                    "CPU": 64,
                    "GPU": 80
                },
                "CPU": 16,
                "Price": 10.21,
                "ChargeType": [
                    "Postpay",
                    "Spot"
                ]
            }
        ]
EOF
    )"
    compute_spec="$(jq 'sort_by(.Price)' <<< "${compute_spec}")"
    echo "${compute_spec}"
}

invoke_action() {
    # Call the API for the specified action
    #
    # Params:
    # * API action specification in JSON
    # * (Optional) file containing the base64-encoded login password
    local action_spec="${1:-}"
    local encoded_password_file="${2:-}"
    [[ -z ${action_spec} ]] && { echo 'Parameter error!' >&2; return 1; }

    local request_url
    request_url="$(gen_request_url \
        "${action_spec}" \
        "${encoded_password_file}")"

    local password
    if [[ -n ${encoded_password_file} ]]; then
        password=(--url-query "Password@${encoded_password_file}")
    fi

    local response
    response="$(curl -LsSf \
        --retry 5 --retry-delay 5 --retry-all-errors \
        "${password[@]}" \
        "${request_url}")"

    echo "${response}"
}


#---------------------------------------------------------------------
# CompShare API wrappers
#---------------------------------------------------------------------
check_resource_capacity() {
    # Check if there are enough resources of specified type.
    # See https://www.compshare.cn/docs/gpus/instance/checkcompshareresourcecapacity
    #
    # Params:
    # * GPU type, such as P40, 3090
    # * charge type, such as Postpay, Spot
    # * image ID
    # * zone
    local gpu_type="${1:-}"
    local charge_type="${2:-}"
    local image_id="${3:-}"
    local zone="${4:-}"

    [[ -z ${gpu_type} \
      || -z ${charge_type} \
      || -z ${image_id} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"CheckCompShareResourceCapacity\", \
        \"ChargeType\": \"${charge_type}\", \
        \"CompShareImageId\": \"${image_id}\", \
        \"Disks.0.IsBoot\": true, \
        \"Disks.0.Size\": 100, \
        \"Disks.0.Type\": \"CLOUD_SSD\", \
        \"GpuType\": \"${gpu_type}\", \
        \"MachineType\": \"G\", \
        \"MinimalCpuPlatform\": \"Auto\", \
        \"Region\": \"${region}\", \
        \"Zone\": \"${zone}\" \
    }"

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

create_instance() {
    # Create a VM instance
    # See https://www.compshare.cn/docs/gpus/instance/createcompshareinstance
    #
    # Reponse:
    #   {
    #       "Action": "CreateCompShareInstanceResponse", 
    #       "RetCode": 0, 
    #       "UHostIds": [
    #           "NIdfqvRv"
    #       ]
    #   }
    #
    # Params:
    # * VM name
    # * file containing the base64-encoded login password
    # * GPU type, such as P40, 3090
    # * CPU cores
    # * memory in MB
    # * charge type
    # * image ID
    # * zone
    local vm_name="${1:-}"
    local encoded_password_file="${2:-}"
    local gpu_type="${3:-}"
    local cpu_cores="${4:-}"
    local memory="${5:-}"
    local charge_type="${6:-}"
    local image_id="${7:-}"
    local zone="${8:-}"
    [[ -z ${vm_name} \
      || -z ${encoded_password_file} \
      || -z ${gpu_type} \
      || -z ${cpu_cores} \
      || -z ${memory} \
      || -z ${charge_type} \
      || -z ${image_id} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"CreateCompShareInstance\", \
        \"CPU\": ${cpu_cores}, \
        \"ChargeType\": \"${charge_type}\", \
        \"CompShareImageId\": \"${image_id}\", \
        \"Disks.0.IsBoot\": true, \
        \"Disks.0.Size\": 100, \
        \"Disks.0.Type\": \"CLOUD_SSD\", \
        \"GPU\": 1, \
        \"GpuType\": \"${gpu_type}\", \
        \"MachineType\": \"G\", \
        \"Memory\": ${memory}, \
        \"Name\": \"${vm_name}\", \
        \"Region\": \"${region}\", \
        \"Zone\": \"${zone}\" \
    }"
    
    local response
    response="$(invoke_action \
        "${action_spec}" \
        "${encoded_password_file}")"
    echo "${response}"
}

describe_available_instance_types() {
    # Get the list of all instance types provided in the zone.
    # See https://www.compshare.cn/docs/gpus/instance/describeavailablecompshareinstancetypes
    #
    # Params:
    # * a JSON object containing the parameters for the API with the
    #   following keys required:
    #   + Zone
    #
    # Return looks like:
    # {
    #     "RetCode": 0,
    #     "AvailableInstanceTypes": [
    #         {
    #             "Name": "5090",
    #             "Status": "Normal",
    #             "MachineSizes": [
    #                 {
    #                     "Gpu": 1,
    #                     "Collection": [
    #                         {
    #                             "Cpu": 16,
    #                             "Memory": [96]
    #                         }
    #                     ]
    #                 },
    #                 {
    #                     "Gpu": 2,
    #                     "Collection": [
    #                         {
    #                             "Cpu": 32,
    #                             "Memory": [192]
    #                         }
    #                     ]
    #                 }
    #             ],
    #             "GraphicsMemory": {
    #                 "Value": 32,
    #                 "Rate": 3
    #             },
    #             "MachineClass": "GPU",
    #             "InstanceType": "uhost",
    #             "ParentType": "G"
    #         },
    #         {
    #             "Name": "4090",
    #             "Status": "Normal",
    #             ...
    #         },
    #         ...
    #     ]
    # }
    local params="${1:-}"
    local zone
    zone="$(jq -r '.Zone // empty' <<< "${params}")"

    [[ -z ${params} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local action_spec='{\
        "Action": "DescribeAvailableCompShareInstanceTypes" \
    }'
    action_spec="$(update_json "${action_spec}" "${params}")"

    if jq -e 'has("Region") | not' <<< "${params}" > /dev/null; then
        local region="{\"Region\": \"${zone%-*}\"}"
        action_spec="$(update_json "${action_spec}" "${region}")"
    fi

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

describe_images() {
    # Get a list of system images
    # See https://www.compshare.cn/docs/gpus/image/describecompshareimages
    #
    # Params:
    # * zone
    local zone="${1:-}"

    [[ -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"DescribeCompShareImages\", \
        \"ImageType\": \"System\", \
        \"Region\": \"${region}\", \
        \"Zone\": \"${zone}\" \
    }"

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

describe_instance() {
    # Get the list of VMs
    # See https://www.compshare.cn/docs/gpus/instance/describecompshareinstance
    local action_spec="{\"Action\": \"DescribeCompShareInstance\"}"
    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

describe_zones() {
    # Get the list of zones
    # See https://www.compshare.cn/docs/gpus/instance/describecompsharesupportzone
    #
    # Return looks like:
    #
    # {
    #     "RetCode": 0,
    #     "ZoneInfo": [
    #         {
    #             "Region": "cn-wlcb",
    #             "RegionId": 1000039,
    #             "Zone": "cn-wlcb-01",
    #             "ZoneId": 10027,
    #             "IsPod": false,
    #             "UnsupportedImageTypes": []
    #         },
    #         {
    #             "Region": "cn-wlcb",
    #             "RegionId": 1000039,
    #             "Zone": "cn-wlcb-03",
    #             "ZoneId": 10033,
    #             "IsPod": true,
    #             "UnsupportedImageTypes": [
    #                 "System",
    #                 "Other"
    #             ]
    #         },
    #     ]
    # }

    local action_spec="{\"Action\": \"DescribeCompShareSupportZone\"}"
    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

get_instance_price() {
    # Get the price for creating the instance.
    # See https://www.compshare.cn/docs/gpus/instance/getcompshareinstanceprice
    #
    # Params:
    # * GPU type
    # * CPU cores
    # * memory in MB
    # * zone
    local gpu_type="${1:-}"
    local cpu_cores="${2:-}"
    local memory="${3:-}"
    local zone="${4:-}"

    [[ -z ${gpu_type} \
      || -z ${cpu_cores} \
      || -z ${memory} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"GetCompShareInstancePrice\", \
        \"Cpu\": ${cpu_cores}, \
        \"Gpu\": 1, \
        \"GpuType\": \"${gpu_type}\", \
        \"Memory\": ${memory}, \
        \"Region\": \"${region}\", \
        \"Zone\": \"${zone}\" \
    }"

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

get_project_list() {
    # Get the list of projects
    # See https://docs.ucloud.cn/api/uaccount-api/get_project_list
    local action_spec="{\"Action\": \"GetProjectList\"}"
    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

stop_instance() {
    # Shutdown the specified VM
    # See https://www.compshare.cn/docs/gpus/instance/stopcompshareinstance
    #
    # Params:
    # * VM ID
    # * zone
    local vm_id="${1:-}"
    local zone="${2:-}"
    [[ -z ${vm_id} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"StopCompShareInstance\", \
        \"Region\": \"${region}\", \
        \"UHostId\": \"${vm_id}\", \
        \"Zone\": \"${zone}\" \
    }"

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

terminate_instance() {
    # Delete the specified VM
    # See https://www.compshare.cn/docs/gpus/instance/terminatecompshareinstance
    #
    # NOTE: The VM must be shut down before deletion
    #
    # Params:
    # * VM ID
    # * zone
    local vm_id="${1:-}"
    local zone="${2:-}"
    [[ -z ${vm_id} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"TerminateCompShareInstance\", \
        \"Region\": \"${region}\", \
        \"ReleaseUDisk\": true, \
        \"UHostId\": \"${vm_id}\", \
        \"Zone\": \"${zone}\" \
    }"

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}

update_stop_scheduler() {
    # Set/update scheduler to stop VM
    # See https://www.compshare.cn/docs/gpus/instance/updatecompsharestopscheduler
    #
    # Params:
    # * VM ID
    # * Time to stop: seconds since the Epoch (1970-01-01 00:00 UTC),
    #   in 3 hours by default
    local vm_id="${1:-}"
    local stop_time="${2:-}"
    local zone="${3:-}"

    [[ -z ${vm_id} \
      || -z ${zone} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    [[ -z ${stop_time} ]] \
        && stop_time="$(date --date='3 hours' '+%s')"

    local projects
    projects="$(api_call_retry get_project_list)"
    local project_id
    project_id="$(jq -r '
        .ProjectSet[]
        | select(.IsDefault)
        | .ProjectId' <<< "${projects}")"

    local region="${zone%-*}"
    local action_spec="{\
        \"Action\": \"UpdateCompShareStopScheduler\", \
        \"ProjectId\": \"${project_id}\", \
        \"Region\": \"${region}\", \
        \"SchedulerStopTime\": ${stop_time}, \
        \"UHostId\": \"${vm_id}\", \
        \"Zone\": \"${zone}\" \
    }"

    local response
    response="$(invoke_action "${action_spec}")"
    echo "${response}"
}


#---------------------------------------------------------------------
# CompShare API utils
#---------------------------------------------------------------------
allocate_vm() {
    # Create a VM satisfying the requirements from available specs.
    #
    # Params:
    # * VM name
    # * file containing the base64-encoded login password
    # * requirements in JSON, for example
    #   + {"GPUType":"!2080,P40","Memory":{"GPU":10,"CPU":9}}
    #     - It means the GPUType should not be 2080 and P40,
    #       GPU memory should be greater than or equal to 10GB
    #       and CPU 9GB.
    #   + {"GPUType":"2080,P40"}
    #     - It means the GPUType should be 2080 or P40.
    local vm_name="${1:-}"
    local encoded_password_file="${2:-}"
    local requirements="${3:-}"
    [[ -z ${vm_name} \
      || -z ${encoded_password_file} \
      || ! -f ${encoded_password_file} \
      || -z ${requirements} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    echo "Allocating a new VM named ${vm_name} ..."
    local compute_spec
    compute_spec="$(get_compute_spec)"

    local num_computes
    num_computes="$(jq 'length' <<< "${compute_spec}")"
    local index
    for ((index=0; index<"${num_computes}"; index++)); do
        local compute
        compute="$(jq -c ".[${index}]" <<< "${compute_spec}")"
        echo "* Trying spec: ${compute}"

        # Check if the compute satisfy requirements
        local reqt
        reqt="$(jq -e 'del(.ChargeType)' <<< "${requirements}")"
        if jq -e 'length != 0' <<< "${reqt}" > /dev/null; then
            local match
            match="$(check_vm_requirement "${compute}" "${reqt}")"
            if [[ "${match}" != 'true' ]]; then
                echo '  + Requirements mismatch.'
                continue
            fi
        fi

        local gpu_type
        gpu_type="$(jq -r '.GPUType' <<< "${compute}")"

        local cpu_cores
        cpu_cores="$(jq '.CPU' <<< "${compute}")"

        local memory
        memory="$(jq '.Memory.CPU * 1024' <<< "${compute}")"

        local available_charge_type
        available_charge_type="$(jq '.ChargeType' <<< "${compute}")"

        local required_charge_types
        if jq -e 'has("ChargeType")' <<< "${requirements}" \
            > /dev/null; then
            readarray -t required_charge_types < \
                <(jq -r '.ChargeType.[]' <<< "${requirements}")
        else
            required_charge_types=('Spot' 'Postpay')
        fi
        local charge_type
        for charge_type in "${required_charge_types[@]}"; do
            if jq -e "
                map((. | ascii_downcase) == \"${charge_type@L}\")
                | any" <<< "${available_charge_type}" > /dev/null
            then
                echo "  + Trying charge type: ${charge_type} ..."
                # Try to create the VM 2 times
                api_call_retry 1 create_instance \
                    "${vm_name}" \
                    "${encoded_password_file}" \
                    "${gpu_type}" \
                    "${cpu_cores}" \
                    "${memory}" \
                    "${charge_type}" \
                    "${image_id:-${COMPSHARE_IMAGE_UBUNTU2404}}" \
                    "${zone:-${COMPSHARE_ZONE_CHINA_NORTH_2A}}" \
                    > /dev/null && return
            fi
        done
    done
    echo 'No available resources!' >&2
    return 1
}

api_call_retry() {
    # Run the API call in "$@" and retry "$1" times
    # (5 by default) on failure.
    #
    # Params:
    # * (optional) number of attempts
    local num_attempts=5
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        num_attempts="$1"
        shift
    fi

    local delay=5
    local attempt=1
    local response
    while true; do
        response="$("$@")"

        local retcode
        retcode="$(jq '.RetCode' <<< "${response}")"
        if [[ ${retcode} == 0 ]]; then
            break
        fi
        echo "ERROR: ${response}" >&2
        if ((attempt >= num_attempts)); then
            echo "ERROR: API call failed after ${num_attempts} attempts." >&2
            return 1
        fi
        echo "Attempt ${attempt} failed! Retrying in ${delay} seconds ..." >&2
        sleep "${delay}"
        ((attempt++))
    done

    echo "${response}"
}

get_gpu_type_combinations() {
    # Returns available GPU types in the following format
    #
    #   {"GpuType":"5090"}
    #   {"GpuType":"4090"}
    #   {"GpuType":"4090_48G"}
    #   {"GpuType":"2080Ti"}
    #
    # Params:
    # * a JSON object of input variables.
    #   + For example,
    #
    #     {
    #         "GpuType": [
    #             "3080Ti",
    #             "3090"
    #         ],
    #         "Zone": [
    #             "cn-wlcb-01",
    #             "cn-sh2-02"
    #         ]
    #     }
    #
    # * parameters for querying the API
    local input_vars="${1:-}"
    local params="${2:-}"

    [[ -z ${input_vars} \
      && -z ${params} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    if jq -e 'has("GpuType")' <<< "${input_vars}" \
        > /dev/null; then
        jq -c '.GpuType[] | {"GpuType": .}' \
                <<< "${input_vars}"
    else
        local gpu_list
        gpu_list="$(api_call_retry \
            describe_available_instance_types "${params}")"
        jq -c '.AvailableInstanceTypes[]
            | select(.Status == "Normal")
            | {"GpuType": .Name}' <<< "${gpu_list}"
    fi
}

get_zone_combinations() {
    # Returns available zones in the following format
    #
    #   {"Zone":"cn-wlcb-01"}
    #   {"Zone":"cn-sh2-02"}
    #
    # Params:
    # * a JSON object of input variables.
    #   + For example,
    #
    #     {
    #         "GpuType": [
    #             "3080Ti",
    #             "3090"
    #         ],
    #         "Zone": [
    #             "cn-wlcb-01",
    #             "cn-sh2-02"
    #         ]
    #     }
    local input_vars="${1:-}"

    if jq -e 'has("Zone")' <<< "${input_vars}" \
        > /dev/null; then
        jq -c '.Zone[] | {"Zone": .}' <<< "${input_vars}"
    else
        local zone_list
        zone_list="$(api_call_retry describe_zones)"
        jq -c '.ZoneInfo[]
            | select(.IsPod | not)
            | .Zone
            | {"Zone": .}' <<< "${zone_list}"
    fi
}

get_vm_info() {
    # Get VM info
    #
    # Returns:
    # * VM ID
    # * SSH destination, in the format like `user@ip_address`
    #
    # Params:
    # * VM name
    local vm_name="${1:-}"

    [[ -z ${vm_name} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    echo "Getting info of the VM ..." >&2
    local response
    response="$(api_call_retry describe_instance)"

    local vm_info
    vm_info="$(jq "
        .UHostSet.[]
        | select(.Name == \"${vm_name}\")" \
        <<< "${response}")"
    [[ -z ${vm_info} ]] \
        && { echo "No VM named ${vm_name}" >&2; return; }
    
    local vm_id
    vm_id="$(jq -r '.UHostId' <<< "${vm_info}")"

    local ssh_dest
    ssh_dest="$(jq -r '.SshLoginCommand' <<< "${vm_info}" \
        | cut -d ' ' -f 2)"

    echo "${vm_id}"
    echo "${ssh_dest}"
}

get_vm_state() {
    # Get the VM state.
    #
    # Params:
    # * VM name
    local vm_name="${1:-}"

    [[ -z ${vm_name} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    echo "Getting info of the VM ..." >&2
    local response
    response="$(api_call_retry describe_instance)"

    local vm_info
    vm_info="$(jq "
        .UHostSet.[]
        | select(.Name == \"${vm_name}\")" \
        <<< "${response}")"
    [[ -z ${vm_info} ]] \
        && { echo "No VM named ${vm_name}" >&2; return; }

    local vm_state
    vm_state="$(jq -r '.State' <<< "${vm_info}")"

    echo "${vm_state}"
}

wait_for_vm_to_stop() {
    # Check and wait for the VM to stop.
    #
    # Params:
    # * VM name
    local vm_name="${1:-}"

    [[ -z ${vm_name} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    echo 'Waiting for the VM to stop ...'
    sleep 5
    local count=0
    local vm_state
    until vm_state="$(get_vm_state "${vm_name}")" \
        && [[ ${vm_state} == 'Stopped'  ]]
    do
        # Set timeout to 5 + 5 * 60 = 305 seconds
        [[ "${count}" -gt 60 ]] \
            && { echo 'Time out!' >&2; return 1; }
        count=$((count + 1))
        echo '* Still waiting ...'
        sleep 5
    done
}
