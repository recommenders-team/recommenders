#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# The following environment variables must be set:
# * COMPSHARE_PRIVATE_KEY
# * COMPSHARE_PUBLIC_KEY
# * COMPSHARE_SPEC_FILE
######################################################################
# Utils used by other functions
######################################################################
get_compute_spec() {
    # Get the specification for all computes available in the
    # CompShare specification file, which is in the form below:
    #
    #   {
    #       "Compute": [  # available computes ranked by their hourly price
    #           {
    #               "GPUType": "P40",
    #               "Price": 0.38,
    #               ...
    #           },
    #           ...
    #       ]
    #       "Action": {
    #           "DescribeCompShareInstance": {
    #               ...
    #           },
    #           ...
    #       }
    #   }
    local compute_spec
    # COMPSHARE_SPEC_FILE is not set directly in the script
    compute_spec="$(jq '.Compute | sort_by(.Price)' "${COMPSHARE_SPEC_FILE}")"
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
    # COMPSHARE_SPEC_FILE is not set directly in the script
    action_template="$(jq ".Action.${action}" "${COMPSHARE_SPEC_FILE}")"

    # COMPSHARE_PUBLIC_KEY is not set directly in the script
    action_template="$(echo "${action_template}" \
        | jq ".PublicKey = \"${COMPSHARE_PUBLIC_KEY}\"")"

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
    params="$(echo "${action_spec}" \
        | jq -r 'to_entries | map("\(.key)=\(.value)") | join("&")')"
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


######################################################################
# Functions for CompShare APIs
######################################################################
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
    # * Time to stop: seconds since the Epoch (1970-01-01 00:00 UTC)
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


######################################################################
# Other utils
######################################################################
check_vm_requirement() {
    # Check if the VM specification match the requirements.
    #
    # Params:
    # * VM specification in JSON
    # * requirements in JSON
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

allocate_vm() {
    # Create a VM with random names and password from available types
    #
    # Params:
    # * VM name
    # * file containing the base64-encoded login password
    # * whether the VM will be used for more than half an hour
    #   + Unit tests require less than half hour
    #   + Nightly tests require more than half hour
    # * (optional) requirements in JSON, for example
    #   + {"GPUType":"!2080,P40","Memory":{"GPU":10,"CPU":9}}
    #     - It means the GPUType should not be 2080 and P40,
    #       GPU memory should be greater than or equal to 10GB
    #       and CPU 9GB.
    #   + {"GPUType":"2080,P40"}
    #     - It means the GPUType should be 2080 or P40.
    local vm_name="${1:-}"
    local encoded_password_file="${2:-}"
    local more_than_half_hour="${3:-}"
    local requirements="${4:-}"
    [[ -z "${vm_name}" \
      || -z "${encoded_password_file}" \
      || ! -f "${encoded_password_file}" \
      || -z "${more_than_half_hour}" ]] && return 1

    echo "Allocating a new VM named ${vm_name} ..." >&2
    local compute_spec
    compute_spec="$(get_compute_spec)"

    local charge_type_list=('Spot' 'Postpay')
    if [[ "${more_than_half_hour}" == 'yes' ]]; then
        charge_type_list=('Postpay')
    fi

    local num_computes
    num_computes="$(echo "${compute_spec}" | jq 'length')"
    for ((i=0; i<"${num_computes}"; i++)); do
        local compute
        compute="$(echo "${compute_spec}" | jq -c ".[${i}]")"
        echo "* Trying spec: ${compute}" >&2

        if [[ -n "${requirements}" ]]; then
            local match
            match="$(check_vm_requirement \
                "${compute}" \
                "${requirements}")"
            if [[ "${match}" != 'true' ]]; then
                echo '  + Requirements mismatch.' >&2
                continue
            fi
        fi

        local gpu_type
        gpu_type="$(echo "${compute}" | jq -r '.GPUType')"

        local cpu_cores
        cpu_cores="$(echo "${compute}" | jq '.CPU')"

        local memory
        memory="$(echo "${compute}" | jq '.Memory.CPU * 1024')"

        for index in "${!charge_type_list[@]}"; do
            charge_type="${charge_type_list[${index}]}"
            local response
            for ((j=0; j<3; j++)); do
                response=$(create_instance \
                    "${vm_name}" \
                    "${encoded_password_file}" \
                    "${gpu_type}" \
                    "${cpu_cores}" \
                    "${memory}" \
                    "${charge_type}")

                local retcode
                retcode="$(echo "${response}" | jq '.RetCode')"
                if [[ ${retcode} == 0 ]]; then
                    return
                fi
                echo "ERROR: ${response}" >&2
                sleep 5
            done
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
    local retcode
    local count=0
    while true; do
        response="$(describe_instance)"
        retcode="$(echo "${response}" | jq '.RetCode')"
        if [[ ${retcode} == 0 ]]; then
            break
        fi
        echo "* Failed to get the info: ${response}." >&2
        count=$((count + 1))
        if [[ $count -lt 5 ]]; then
            sleep $(( (count+1) * 5 ))
            echo '* Trying again ...' >&2
        else
            return 1
        fi
    done

    local vm_info
    vm_info="$(echo "${response}" \
        | jq ".UHostSet.[] | select(.Name == \"${vm_name}\")")"
    [[ -z "${vm_info}" ]] && return 1
    
    local vm_id
    vm_id="$(echo "${vm_info}" | jq -r '.UHostId')"

    local ssh_dest
    ssh_dest="$(echo "${vm_info}" \
        | jq -r '.SshLoginCommand' | cut -d ' ' -f 2)"

    echo "${vm_id}"
    echo "${ssh_dest}"
}

wait_for_vm_to_be_available() {
    # Check and wait for the VM being available
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
        || (echo "${ssh_response}" | grep -iq 'permission')
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
    local count=0
    until sshpass -e ssh-copy-id \
        -i "${key_file}.pub" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}"
    do
        [[ "${count}" -gt 5 ]] && return 1
        count=$((count + 1))
        sleep 5
        echo '  + Trying again ...' >&2
    done

    echo '* Disabling SSH password authentication ...' >&2
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}" "\
            sudo sed -i -E 's/^[[:space:]#]*PasswordAuthentication.*/PasswordAuthentication no/' ${sshd_config}; \
            sudo systemctl reload ssh"
}
