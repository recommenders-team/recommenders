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
script_path="$(realpath -- "${BASH_SOURCE[0]}")"
script_dir="$(dirname -- "${script_path}")"

# Source common utils
source "${script_dir}/../utils.sh"


#---------------------------------------------------------------------
# Utils used by other CompShare API wrappers and utils
#---------------------------------------------------------------------
add_region() {
    # Extract the region from the zone in params.
    #
    # Params:
    # * a JSON object containing the parameters for the API with the
    #   following keys required:
    #   + Zone
    local params="${1:-}"

    [[ -z ${params} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    if jq -e 'has("Zone")
        and ((has("Region") and (.Region | not))
            or (has("Region") | not))' <<< "${params}" > /dev/null
    then
        local zone
        zone="$(jq -r '.Zone' <<< "${params}")"
        local region="{\"Region\": \"${zone%-*}\"}"
        params="$(update_json "${params}" "${region}")"
    fi
    echo "${params}"
}

check_vm_requirement() {
    # Check if the VM specification match the requirements.
    #
    # Params:
    # * VM specification in JSON
    # * requirements in JSON, for example
    #   + {"GPUType":"!2080,P40","Memory":10240,"GraphicsMemory":10240}
    #     - It means the GPUType should not be 2080 and P40,
    #       GPU memory should >= 10240MB
    #       and CPU 10240MB.
    #   + {"GPUType":"2080,P40"}
    #     - It means the GPUType should be 2080 or P40.
    #   + {"GPUType":["2080","P40"],"ChargeType":"Spot"}
    #     - It means the GPUType should be 2080 or P40,
    #       ChargeType should be Spot.
    local specs="${1:-}"
    local requirements="${2:-}"

    local match
    match="$(jq -n --argjson specs "${specs}" \
        --argjson reqs "${requirements}" \
        'def equalstr($a; $b):
            if ($a | startswith(" ")) then
                equalstr(($a | ltrimstr(" ")); $b)
            elif ($a | endswith(" ")) then
                equalstr(($a | rtrimstr(" ")); $b)
            else
                $a == $b
            end;
        def compareitem($reqs; $specs; $p):
            ($reqs | getpath($p)) as $r
            | ($specs | getpath($p)) as $s
            | ($r | type) as $r_type
            | if $r_type == "string" and $s then
                $r
                | if startswith("!") then
                    $r | ltrimstr("!") | split(",")
                    | reduce .[] as $ri
                        (true; . and (equalstr($ri; $s) | not))
                else
                    $r | split(",")
                    | reduce .[] as $ri
                        (false; . or equalstr($ri; $s))
                end
                | if . then . else
                    debug("\($p[0]) requires \($r), but got \($s)")
                end
            elif $r_type == "number" and $s then
                $r <= $s
                | if . then . else
                    debug("\($p[0]) must >= \($r), but got \($s)")
                end
            elif $r_type == "array" and $s then
                reduce $r.[] as $ri (false; . or $ri == $s)
                | if . then . else
                    debug("\($p[0]) requires \($r), but got \($s)")
                end
            else
                true
            end;
        $reqs
        | [path(..) | select(length | . == 1)]
        | reduce .[] as $p
            (true; . and compareitem($reqs; $specs; $p))')"

    echo "${match}"
}

gen_action_digest() {
    # Generate the digest for the action requrest parameters
    # See https://docs.ucloud.cn/api/summary/signature
    # 
    # Params:
    # * API action specification in JSON
    local action_spec="${1:-}"
    [[ -z ${action_spec} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

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
            | join("")' <<< "${action_spec}" \
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
    local action_spec="${1:-}"

    [[ -z ${action_spec} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    # COMPSHARE_PUBLIC_KEY is an environment variable.
    action_spec="$(jq ".PublicKey = \"${COMPSHARE_PUBLIC_KEY}\"" \
        <<< "${action_spec}")"

    local digest
    digest="$(gen_action_digest "${action_spec}")"
    local params
    params="$(jq -r '
        to_entries
        | map("\(.key)=\(.value | @uri)")
        | join("&")' <<< "${action_spec}")"
    echo "https://api.compshare.cn/?${params}&Signature=${digest}"
}

invoke_action() {
    # Call the API for the specified action
    #
    # Params:
    # * action
    # * a JSON object containing the parameters for the action API
    # * the config.yml file containing the action specification
    local config_yml="${1:-}"
    local action="${2:-}"
    local parameters="${3:-{\}}"

    [[ -z ${action} \
      || ! -f ${config_yml} ]] \
      && { echo 'Parameter error!' >&2; return 1; }
    
    local action_spec
    action_spec="$(yq -o json < "${config_yml}" \
        | jq -c ".action_specs.${action}")"

    action="{\"Action\": \"${action}\"}"
    parameters="$(update_json "${parameters}" "${action}")"
    parameters="$(add_region "${parameters}")"

    local required_params
    readarray -t required_params < \
        <(jq -rc 'to_entries?
            | map(select(.value | not))
            | map(.key)
            | .[]' <<< "${action_spec}")

    local param_index
    for param_index in "${!required_params[@]}"; do
        local param="${required_params[${param_index}]}"
        if jq -e "has(\"${param}\") | not" <<< "${parameters}" \
            > /dev/null; then
            echo "Parameter '${param}' missing!" >&2
            return 1
        fi
    done

    action_spec="$(update_json "${action_spec}" "${parameters}")"
    
    local request_url
    request_url="$(gen_request_url "${action_spec}")"

    local response
    response="$(curl -LsSf \
        --retry 5 --retry-delay 5 --retry-all-errors \
        "${request_url}")"

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
    # * requirements in JSON, for example
    #   + {"GPUType":"!2080,P40","Memory":10240,"GraphicsMemory":10240}
    #     - It means the GPUType should not be 2080 and P40,
    #       GPU memory should >= 10240MB
    #       and CPU 10240MB.
    #   + {"GPUType":"2080,P40"}
    #     - It means the GPUType should be 2080 or P40.
    #   + {"GPUType":["2080","P40"],"ChargeType":"Spot"}
    #     - It means the GPUType should be 2080 or P40,
    #       ChargeType should be Spot.
    # * a JSON object of input variables such as Zone and GpuType
    #   + For example,
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
    #             "Postpay",
    #             "Spot"
    #         ]
    #     }
    local vm_name="${1:-}"
    local requirements="${2:-}"
    local input_vars="${3:-}"
    local config_yml="${4:-}"

    [[ -z ${vm_name} \
      || -z ${requirements} ]] \
      && { echo 'Parameter error!' >&2; return 1; }

    echo 'Getting available instance info ...'
    local compute_list
    readarray -t compute_list < \
        <(get_available_required_computes \
            "${input_vars}" "${config_yml}")

    local compute_index
    for compute_index in "${!compute_list[@]}"; do
        local compute
        compute="${compute_list[${compute_index}]}"
        compute="$(jq -c 'del(.Price)' <<< "${compute}")"
        echo "Trying to create a VM named ${vm_name} of ${compute%,\"GraphicsMemory*} ..."

        if jq -e 'length != 0' <<< "${requirements}" > /dev/null; then
            local match
            match="$(check_vm_requirement "${compute}" "${requirements}")"
            if [[ "${match}" != 'true' ]]; then
                continue
            fi
        fi

        vm_name="{\"Name\": \"${vm_name}\"}"
        compute="$(update_json "${compute}" "${vm_name}")"
        api_call_retry 1 invoke_action "${config_yml}" \
            'CreateCompShareInstance' "${compute}" > /dev/null \
            && return
    done
    echo 'No available required resources!' >&2
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

get_available_required_computes() {
    # Returns the infos of available required computes sorted by price
    # in the following format including the prices:
    #
    # {"ChargeType":"u","GpuType":"x","GraphicsMemory":32*1024,"Zone":"y","Region":"z","CompShareImageId":"w","Cpu":16,"Gpu":1,"Memory":65536,"Price":1.5}
    # {"ChargeType":"u","GpuType":"x","GraphicsMemory":32*1024,"Zone":"y","Region":"x","CompShareImageId":"w","Cpu":16,"Gpu":1,"Memory":96256,"Price":1.6}
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
    #         ],
    #         "ChargeType": [
    #             "Postpay"
    #         ]
    #     }
    local input_vars="${1:-}"
    local config_yml="${2:-}"
    local compute_list='[]'

    echo '* Getting available zones ...' >&2
    local zone_list
    readarray -t zone_list < \
        <(get_available_required_zones \
            "${input_vars}" "${config_yml}")

    local zone_index
    for zone_index in "${!zone_list[@]}"; do
        local zone="${zone_list[${zone_index}]}"

        echo "* Getting available GPUs in ${zone%%,*} ..." >&2
        local gpu_list
        readarray -t gpu_list < \
            <(get_available_required_gpu_types \
                "${input_vars}" "${zone}" "${config_yml}")

        local gpu_index
        for gpu_index in "${!gpu_list[@]}"; do
            local gpu="${gpu_list[${gpu_index}]}"

            # Add the image ID.
            local image_id='{
                "CompShareImageId": "compshareImage-12rjyhwynazd"
            }'
            gpu="$(update_json "${gpu}" "${image_id}")"

            local spec_list
            readarray -t spec_list < \
                <(get_available_required_gpu_specs \
                    "${input_vars}" "${gpu}" "${config_yml}")

            local spec_index
            for spec_index in "${!spec_list[@]}"; do
                local spec="${spec_list[${spec_index}]}"

                local compute
                compute="$(get_gpu_spec_price "${spec}" "${config_yml}")"
                compute_list="$(jq -nc \
                    --argjson arr "${compute_list}" \
                    --argjson obj "${compute}" \
                    '$arr + [$obj]')"
            done
        done
    done

    local compute_list
    compute_list="$(jq -c 'sort_by(.Price) | .[]' \
        <<< "${compute_list}")"
    echo "${compute_list}"
}

get_available_required_gpu_specs() {
    # Returns specifications of the available required GPUs in the
    # following format:
    #
    #   {"ChargeType":"Postpay","GpuType":"z","GraphicsMemory":32*1024,"Zone":"y","Region":"x","CompShareImageId":"w","Cpu":16,"Gpu":1,"Memory":64*1024}
    #   {"ChargeType":"Spot","GpuType":"z","GraphicsMemory":32*1024,"Zone":"y","Region":"x","CompShareImageId":"w","Cpu":16,"Gpu":1,"Memory":128*1024}
    #
    # Params:
    # * a JSON object of input variables that may specified the
    #   the required charge types.
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
    #         ],
    #         "ChargeType": [
    #             "Postpay"
    #         ]
    #     }
    #
    # * parameters for querying the API
    #   + For example,
    #     {"GpuType":"z","GraphicsMemory":32*1024,"Zone":"y","Region":"x","CompShareImageId":"w"}
    # * path to config_yml
    local input_vars="${1:-}"
    local params="${2:-}"
    local config_yml="${3:-}"

    [[ -z ${params} ]] && { echo 'Parameter error!' >&2; return 1; }

    # Get the available required charge types.
    local charge_type_list='["Spot", "Postpay"]'
    charge_type_list="$(get_available_required_items \
        "${charge_type_list}" "${input_vars}" 'ChargeType')"
    readarray -t charge_type_list < \
        <(jq -c '.[] | {"ChargeType": .}' <<< "${charge_type_list}")

    local index
    for index in "${!charge_type_list[@]}"; do
        # Add charge type to params
        local charge_type="${charge_type_list[${index}]}"
        params="$(update_json "${params}" "${charge_type}")"

        # Get the list of instance types with enough resources. 
        local spec_list
        spec_list="$(api_call_retry invoke_action "${config_yml}" \
            'CheckCompShareResourceCapacity' "${params}")"
        jq -c --argjson chargetype "${charge_type}" \
            --argjson params "${params}" \
            '.Specs[]
            | select(.ResourceEnough and .Gpu == 1)
            | $chargetype + $params
              + {Cpu, Gpu, "Memory": (.Mem * 1024)}' \
            <<< "${spec_list}"
    done
}

get_available_required_gpu_types() {
    # Returns available required GPU types in the following format:
    #
    #   {"GpuType":"5090","GraphicsMemory":32*1024,"Zone":"cn-wlcb-01","Region":"cn-wlcb"}
    #   {"GpuType":"4090","GraphicsMemory":24*1024,"Zone":"cn-wlcb-01","Region":"cn-wlcb"}
    #   {"GpuType":"4090_48G","GraphicsMemory":48*1024,"Zone":"cn-wlcb-01","Region":"cn-wlcb"}
    #   {"GpuType":"2080Ti","GraphicsMemory":11*1024,"Zone":"cn-wlcb-01","Region":"cn-wlcb"}
    #
    # Params:
    # * a JSON object of input variables that may specifiy the
    #   required GPU Types.
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
    #         ],
    #         "ChargeType": [
    #             "Postpay"
    #         ]
    #     }
    #
    # * parameters for querying the API
    #   + For example, {"Zone":"cn-wlcb-01","Region":"cn-wlcb"}
    # * path to config.yml
    local input_vars="${1:-}"
    local params="${2:-}"
    local config_yml="${3:-}"

    [[ -z ${params} ]] && { echo 'Parameter error!' >&2; return 1; }

    # Get the list of available GPU types in the format like
    #
    #   ["3080","4090","5090"]
    local gpu_types
    gpu_types="$(api_call_retry invoke_action "${config_yml}" \
        'DescribeAvailableCompShareInstanceTypes' "${params}")"
    gpu_types="$(jq -c '[.AvailableInstanceTypes[]
        | select(.Status == "Normal")
        | {"GpuType": .Name,
           "GraphicsMemory": (.GraphicsMemory.Value * 1024)}]' \
        <<< "${gpu_types}")"
    local gpu_list
    gpu_list="$(jq '[ .[].GpuType ]' <<< "${gpu_types}")"

    # Get the available required GPU types.
    gpu_list="$(get_available_required_items \
        "${gpu_list}" "${input_vars}" 'GpuType')"

    jq -c --argjson params "${params}" \
        --argjson gpu_list "${gpu_list}" \
        'map(select([.GpuType] - $gpu_list | length | . == 0))
        | .[]
        | . + $params' <<< "${gpu_types}"
}

get_gpu_spec_price() {
    # Returns the price of the GPU specified in the parameter in the
    # following format:
    #
    #   {"ChargeType":"Postpay","GpuType":"z","GraphicsMemory":32*1024,"Zone":"y","Region":"x","CompShareImageId":"w","Cpu":16,"Gpu":1,"Memory":64*1024,"Price":1.30}
    #
    # Params:
    # * parameters for querying the API
    #   + For example,
    #     {"ChargeType":"Postpay","GpuType":"z","GraphicsMemory":32*1024,"Zone":"y","Region":"x","CompShareImageId":"w","Cpu":16,"Gpu":1,"Memory":64*1024}
    local params="${1:-}"
    local config_yml="${2:-}"

    [[ -z ${params} ]] && { echo 'Parameter error!' >&2; return 1; }

    local price
    price="$(api_call_retry invoke_action "${config_yml}" \
        'GetCompShareInstancePrice' "${params}")"
    jq -c --argjson params "${params}" \
        '.PriceDetails[0]
        | $params + {"Price": .Instance}' \
        <<< "${price}"
}

get_available_required_zones() {
    # Returns available requried zones in the following format:
    #
    #   {"Zone":"cn-wlcb-01","Region":"cn-wlcb"}
    #   {"Zone":"cn-sh2-02","Region":"cn-sh2"}
    #
    # Params:
    # * a JSON object of input variables that may specifiy the
    #   required zones.
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
    #         ],
    #         "ChargeType": [
    #             "Postpay"
    #         ]
    #     }
    local input_vars="${1:-}"
    local config_yml="${2:-}"

    # Get the list of available zones in the format like
    #
    #   ["cn-wlcb-01","cn-sh2-02"]
    local zone_list
    zone_list="$(api_call_retry invoke_action "${config_yml}" \
        'DescribeCompShareSupportZone')"
    zone_list="$(jq -c '[ .ZoneInfo[]
        | select(.IsPod | not)
        | .Zone ]' <<< "${zone_list}")"
    
    # Get the available required zones.
    zone_list="$(get_available_required_items \
        "${zone_list}" "${input_vars}" 'Zone')"

    jq -c '.[]
        | {"Zone": ., "Region": (. | capture("(?<r>.*)-[^-]+").r)}' \
          <<< "${zone_list}"
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
    local config_yml="${2:-}"

    [[ -z ${vm_name} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    local response
    response="$(api_call_retry invoke_action "${config_yml}" \
        'DescribeCompShareInstance')"

    local vm_info
    vm_info="$(jq "
        .UHostSet.[]
        | select(.Name == \"${vm_name}\")" \
        <<< "${response}")"
    [[ -z ${vm_info} ]] \
        && { echo "No VM named ${vm_name}" >&2; return; }

    vm_info="$(jq -c '{
        UHostId, State, Zone, Region, SshLoginCommand, Password
        }' <<< "${vm_info}")"
    echo "${vm_info}"
}

wait_for_vm_to_stop() {
    # Check and wait for the VM to stop.
    #
    # Params:
    # * VM name
    local vm_name="${1:-}"
    local config_yml="${2:-}"

    [[ -z ${vm_name} ]] \
        && { echo 'Parameter error!' >&2; return 1; }

    echo 'Waiting for the VM to stop ...'
    sleep 5
    local count=0
    local vm_state
    until vm_state="$(get_vm_info "${vm_name}" "${config_yml}" \
        | jq -r '.State')" \
        && [[ ${vm_state} == 'Stopped' ]]
    do
        # Set timeout to 5 + 5 * 60 = 305 seconds
        [[ "${count}" -gt 60 ]] \
            && { echo 'Time out!' >&2; return 1; }
        count=$((count + 1))
        echo '* Still waiting ...'
        sleep 5
    done
}
