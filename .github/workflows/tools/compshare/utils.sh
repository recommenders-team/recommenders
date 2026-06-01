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

#---------------------------------------------------------------------
# Utils used by other CompShare API wrappers and utils
#---------------------------------------------------------------------
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
                "Price": 1.88,
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
                "Price": 3.0,
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
                "Price": 5.92,
                "ChargeType": [
                    "Postpay"
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

get_action_template() {
    # Get the specification template for a specific action
    #
    # Params:
    # * API action name
    local action="${1:-}"
    [[ -z "${action}" ]] && return 1

    local action_template
    action_template="$(cat << 'EOF'
        [
            {
                "Action": "CreateCompShareInstance",
                "ChargeType": "Postpay",
                "CompShareImageId": "compshareImage-12rjyhwynazd",
                "Disks.0.IsBoot": true,
                "Disks.0.Size": 100,
                "Disks.0.Type": "CLOUD_SSD",
                "GPU": 1,
                "GPUType": "",
                "MachineType": "G",
                "Memory": 65536,
                "Name": "",
                "Region": "cn-wlcb",
                "Zone": "cn-wlcb-01",
                "CPU": 8
            },
            {
                "Action": "DescribeCompShareInstance"
            },
            {
                "Action": "GetProjectList"
            },
            {
                "Action": "StopCompShareInstance",
                "Region": "cn-wlcb",
                "Zone": "cn-wlcb-01",
                "UHostId": ""
            },
            {
                "Action": "TerminateCompShareInstance",
                "Region": "cn-wlcb",
                "Zone": "cn-wlcb-01",
                "UHostId": "",
                "ReleaseUDisk": true
            },
            {
                "Action": "UpdateCompShareStopScheduler",
                "Region": "cn-wlcb",
                "Zone": "cn-wlcb-01",
                "ProjectId": "org-hmgw4i",
                "UHostId": "",
                "SchedulerStopTime": 1779164372
            }
        ]
EOF
    )"
    action_template="$(jq ".[] | select(.Action == \"${action}\")" \
        <<< "${action_template}")"

    # COMPSHARE_PUBLIC_KEY is not set directly in the script
    action_template="$(jq ".PublicKey = \"${COMPSHARE_PUBLIC_KEY}\"" \
        <<< "${action_template}")"

    echo "${action_template}"
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
    [[ -z "${action_spec}" ]] && return 1

    # Store the spec into a file to hide the password from being visible
    local action_spec_file
    action_spec_file="$(mktemp)"
    echo "${action_spec}" > "${action_spec_file}"
    if [[ -n "${encoded_password_file}" ]]; then
        echo "${action_spec}" \
            | jq --rawfile encoded_password "${encoded_password_file}" \
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
        jq -r 'to_entries | sort | map("\(.key)\(.value)") | join("")' \
            "${action_spec_file}" \
        | tr -d '\n' \
        | cat - <(echo "${COMPSHARE_PRIVATE_KEY}") \
        | tr -d '\n' \
        | sha1sum \
        | head -c 40)"
    rm -rf "${action_spec_file}"

    [[ "${reset_x}" == true ]] && set -x

    echo "${digest}"
}

gen_request_url() {
    # Generate the API request URL using the action specification and
    # the parameter digest
    #
    # Params:
    # * API action name
    # * (Optional) updates for the parameters in JSON
    # * (Optional) file containing the base64-encoded login password
    local action="${1:-}"
    local updates="${2:-}"
    local encoded_password_file="${3:-}"
    [[ -z "${action}" ]] && return 1

    local action_spec
    action_spec="$(get_action_template "${action}")"

    if [[ -n "${updates}" ]]; then
        action_spec="$(update_json "${action_spec}" "${updates}")"
    fi

    local digest
    digest="$(gen_action_digest "${action_spec}" "${encoded_password_file}")"
    local params
    params="$(jq -r 'to_entries | map("\(.key)=\(.value)") | join("&")' \
        <<< "${action_spec}")"
    echo "https://api.compshare.cn/?${params}&Signature=${digest}"
}

invoke_action() {
    # Call the API for the specified action
    #
    # Params:
    # * API action name
    # * (Optional) updates for the parameters in JSON
    # * (Optional) file containing the base64-encoded login password
    local action="${1:-}"
    local updates="${2:-}"
    local encoded_password_file="${3:-}"
    [[ -z "${action}" ]] && return 1

    local request_url
    request_url="$(gen_request_url \
        "${action}" \
        "${updates}" \
        "${encoded_password_file}")"

    local response
    if [[ -n "${encoded_password_file}" ]]; then
        response="$(curl -sSf \
            --url-query "Password@${encoded_password_file}" \
            "${request_url}")"
    else
        response="$(curl -sSf "${request_url}")"
    fi

    echo "${response}"
}


#---------------------------------------------------------------------
# CompShare API wrappers
#---------------------------------------------------------------------
create_instance() {
    # Create a VM instance
    # See https://www.compshare.cn/docs/operation/api/createcompshareinstance
    #
    # Reponse:
    #   {
    #       "Action": "CreateCompShareInstanceResponse", 
    #       "RetCode": 0, 
    #       "UHostIds": [
    #           "NIdfqvRv"
    #       ]
    #   }

    # Params:
    # * VM name
    # * file containing the base64-encoded login password
    # * GPU type, such as P40, 3090
    # * CPU cores
    # * memory in MB
    # * charge type
    local vm_name="${1:-}"
    local encoded_password_file="${2:-}"
    local gpu_type="${3:-}"
    local cpu_cores="${4:-}"
    local memory="${5:-}"
    local charge_type="${6:-}"
    [[ -z "${vm_name}" \
      || -z "${encoded_password_file}" \
      || -z "${gpu_type}" \
      || -z "${cpu_cores}" \
      || -z "${memory}" \
      || -z "${charge_type}" ]] && return 1

    local updates
    updates="{\
        \"Name\": \"${vm_name}\", \
        \"GPUType\": \"${gpu_type}\", \
        \"CPU\": ${cpu_cores}, \
        \"Memory\": ${memory}, \
        \"ChargeType\": \"${charge_type}\"}"
    
    local response
    response="$(invoke_action \
        'CreateCompShareInstance' \
        "${updates}" \
        "${encoded_password_file}")"
    echo "${response}"
}

describe_instance() {
    # Get the list of VMs
    # See https://www.compshare.cn/docs/operation/api/describecompshareinstance

    local response
    response="$(invoke_action 'DescribeCompShareInstance')"

    echo "${response}"
}

get_project_list() {
    # Get the list of projects
    # See https://docs.ucloud.cn/api/uaccount-api/get_project_list
    local response
    response="$(invoke_action 'GetProjectList')"

    echo "${response}"
}

stop_instance() {
    # Shutdown the specified VM
    # See https://www.compshare.cn/docs/operation/api/stopcompshareinstance
    #
    # Params:
    # * VM ID
    local vm_id="${1:-}"
    [[ -z "${vm_id}" ]] && return 1

    local updates
    updates="{\"UHostId\": \"${vm_id}\"}"

    local response
    response="$(invoke_action 'StopCompShareInstance' "${updates}")"
    echo "${response}"
}

terminate_instance() {
    # Delete the specified VM
    # See https://www.compshare.cn/docs/operation/api/terminatecompshareinstance
    #
    # NOTE: The VM must be shut down before deletion
    #
    # Params:
    # * VM ID
    local vm_id="${1:-}"
    [[ -z "${vm_id}" ]] && return 1

    local updates
    updates="{\"UHostId\": \"${vm_id}\"}"

    local response
    response="$(invoke_action 'TerminateCompShareInstance' "${updates}")"
    echo "${response}"
}

update_stop_scheduler() {
    # Set/update scheduler to stop VM
    # See https://www.compshare.cn/docs/gpus/instance/updatecompsharestopscheduler
    #
    # Params:
    # * VM ID
    # * Time to stop: seconds since the Epoch (1970-01-01 00:00 UTC), in 3 hours by default
    local vm_id="${1:-}"
    local stop_time="${2:-}"
    [[ -z "${vm_id}" ]] && return 1
    [[ -z "${stop_time}" ]] && stop_time="$(date --date='3 hours' '+%s')"

    local updates
    updates="{\
        \"UHostId\": \"${vm_id}\", \
        \"SchedulerStopTime\": ${stop_time}}"

    local response
    response="$(invoke_action 'UpdateCompShareStopScheduler' "${updates}")"
    echo "${response}"
}


#---------------------------------------------------------------------
# CompShare API utils
#---------------------------------------------------------------------
allocate_vm() {
    # Create a VM with random names and password from available types
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
    [[ -z "${vm_name}" \
      || -z "${encoded_password_file}" \
      || ! -f "${encoded_password_file}" \
      || -z "${requirements}" ]] && return 1

    echo "Allocating a new VM named ${vm_name} ..." >&2
    local compute_spec
    compute_spec="$(get_compute_spec)"

    local num_computes
    num_computes="$(jq 'length' <<< "${compute_spec}")"
    for ((i=0; i<"${num_computes}"; i++)); do
        local compute
        compute="$(jq -c ".[${i}]" <<< "${compute_spec}")"
        echo "* Trying spec: ${compute}" >&2

        # Check if the compute satisfy requirements
        local reqt
        reqt="$(jq -e 'del(.ChargeType)' <<< "${requirements}")"
        if jq -e 'length != 0' <<< "${reqt}" > /dev/null; then
            local match
            match="$(check_vm_requirement "${compute}" "${reqt}")"
            if [[ "${match}" != 'true' ]]; then
                echo '  + Requirements mismatch.' >&2
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
        mapfile -t required_charge_types < \
            <(jq -rc '.ChargeType.[]' <<< "${requirements}")
        for charge_type in "${required_charge_types[@]}"; do
            if jq -e "map(. == \"${charge_type}\") 
                | any" <<< "${available_charge_type}" > /dev/null
            then
                # Try to create the VM 3 times
                api_call_retry 3 create_instance \
                    "${vm_name}" \
                    "${encoded_password_file}" \
                    "${gpu_type}" \
                    "${cpu_cores}" \
                    "${memory}" \
                    "${charge_type}" > /dev/null && return
            fi
        done
    done
    return 1
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
    [[ -z "${vm_name}" ]] && return 1

    echo "Getting info of the VM ..." >&2
    local response
    response="$(api_call_retry describe_instance)"

    local vm_info
    vm_info="$(jq ".UHostSet.[] | select(.Name == \"${vm_name}\")" \
        <<< "${response}")"
    [[ -z "${vm_info}" ]] && return 1
    
    local vm_id
    vm_id="$(jq -r '.UHostId' <<< "${vm_info}")"

    local ssh_dest
    ssh_dest="$(jq -r '.SshLoginCommand' <<< "${vm_info}" \
        | cut -d ' ' -f 2)"

    echo "${vm_id}"
    echo "${ssh_dest}"
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


######################################################################
# Non Compshare API utils
#
# These utils do not require any preset environment variables.
######################################################################
update_json() {
    # Update a JSON with another JSON
    #
    # Params:
    # * the original JSON
    # * the JSON with all updates
    local original="${1:-}"
    local updates="${2:-}"
    [[ -z "${updates}" || -z "${original}" ]] && return 1

    local res
    res=$(jq -s '
        def update($a; $b):
            ($a | type) as $ta | ($b | type) as $tb |
            if $ta == "object" and $tb == "object" then
                reduce ([$a, $b] | add | keys_unsorted[]) as $k
                    ({}; .[$k] = update($a[$k]; $b[$k]))
            elif $ta == "array" and $tb == "array" then
                $a + $b
            else
                $b // $a
            end;
        reduce .[] as $item (null; update(.; $item))' \
        <(echo "${original}") <(echo "${updates}"))
    echo "${res}"
}

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

wait_for_vm_to_be_available() {
    # Check and wait for the VM being available.
    # It will fail if the VM cannot be accessed after 300 seconds.
    #
    # Params:
    # * SSH destination, in the format like `user@ip_address`
    local ssh_dest="${1:-}"
    [[ -z "${ssh_dest}" ]] && return 1

    echo 'Waiting for the VM to be available ...' >&2
    local count=0
    local ssh_response
    until ssh_response=$(\
        ssh -o BatchMode=yes \
            -o ConnectTimeout=5 \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}" true 2>&1) \
        || grep -iq 'permission' <<< "${ssh_response}"
    do
        # Set timeout to (5 + 5) * 30 = 300 seconds
        [[ "${count}" -gt 30 ]] && return 1
        count=$((count + 1))
        echo '* Still waiting ...' >&2
        sleep 5
    done
}

setup_ssh_key() {
    # Set up SSH key for connection
    #
    # Params:
    # * SSH destination, in the format like `user@ip_address`
    # * file containing the base64-encoded login password
    local ssh_dest="${1:-}"
    local encoded_password_file="${2:-}"
    [[ -z "${ssh_dest}" \
      || -z "${encoded_password_file}" \
      || ! -f "${encoded_password_file}" ]] && return 1

    local key_file="${HOME}/.ssh/id_ed25519"
    local sshd_config="/etc/ssh/sshd_config"

    echo 'Setting up SSH key for login ...' >&2
    echo '* Generating SSH key ...' >&2
    if [[ ! -f "${key_file}" || ! -f "${key_file}.pub" ]]; then
        ssh-keygen -q -t ed25519 -N '' -f "${key_file}"
    fi

    echo '* Deplying SSH key ...' >&2
    local -x SSHPASS
    read -r SSHPASS < <(cat "${encoded_password_file}" | tr -d '\n' | base64 -d) || true
    run_cmd_retry sshpass -e ssh-copy-id \
        -i "${key_file}.pub" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}"

    echo '* Disabling SSH password authentication ...' >&2
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}" "\
            sudo sed -i -E 's/^[[:space:]#]*PasswordAuthentication.*/PasswordAuthentication no/' ${sshd_config}; \
            sudo systemctl reload ssh"
}

apt_install_retry() {
    # Run apt-get install "$@" and retry "$1" times
    # (5 by default) on failure.
    #
    # Params:
    # * (optional) number of attempts
    local num_attempts=5
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        num_attempts="$1"
        shift
    fi

    run_cmd_retry "${num_attempts}" \
        sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
        apt-get install -y "$@"
}

run_cmd_retry() {
    # Run the command in "$@" and retry "$1" times
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
    until "$@"; do
        if ((attempt >= num_attempts)); then
            echo "ERROR: Failed after ${num_attempts} attempts." >&2
            return 1
        fi
        echo "Attempt ${attempt} failed! Retrying in ${delay} seconds ..." >&2
        sleep "${delay}"
        ((attempt++))
    done
}

wait_for_apt_lock() {
    # Wait for processes releasing /var/lib/apt/lists/lock
    while sudo fuser /var/lib/apt/lists/lock 2>/dev/null; do
        echo 'Waiting for processes releasing /var/lib/apt/lists/lock ...' >&2
        sleep 5
    done
}
