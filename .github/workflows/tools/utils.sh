#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Common Utils
######################################################################
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
    # NOTE: This function is only for a single command without
    #       redirection, not for multiple commands.
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

setup_ssh_key() {
    # Set up SSH key for connection
    #
    # Params:
    # * SSH destination, in the format like `user@ip_address`
    # * file containing the base64-encoded login password
    local ssh_dest="${1:-}"
    local encoded_password_file="${2:-}"
    [[ -z ${ssh_dest} \
      || -z ${encoded_password_file} \
      || ! -f ${encoded_password_file} ]] && return 1

    local key_file="${HOME}/.ssh/id_ed25519"
    local sshd_config="/etc/ssh/sshd_config"

    echo 'Setting up SSH key for login ...' >&2
    echo '* Generating SSH key ...' >&2
    if [[ ! -f ${key_file} || ! -f ${key_file}.pub ]]; then
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

update_json() {
    # Update a JSON with another JSON
    #
    # Params:
    # * the original JSON
    # * the JSON with all updates
    local original="${1:-}"
    local updates="${2:-}"
    [[ -z ${updates} || -z ${original} ]] && return 1

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

wait_for_apt_lock() {
    # Wait for processes releasing /var/lib/apt/lists/lock
    while sudo fuser /var/lib/apt/lists/lock 2>/dev/null; do
        echo 'Waiting for processes releasing /var/lib/apt/lists/lock ...' >&2
        sleep 5
    done
}

wait_for_vm_to_be_available() {
    # Check and wait for the VM being available.
    # It will fail if the VM cannot be accessed after 300 seconds.
    #
    # Params:
    # * SSH destination, in the format like `user@ip_address`
    local ssh_dest="${1:-}"
    [[ -z ${ssh_dest} ]] && return 1

    echo 'Waiting for the VM to be available ...' >&2
    # Wait some time for the operation to be completed.
    sleep 5
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

post_create_setup() {
    # Do post-create setup on the VM.
    # 
    # It assumed that there is a configuration file called
    # config.yml in a directory named '$cloud_service' under the
    # tools directory.  In config.yml,
    # * there is a key 'scripts.post_create' with the list of scripts
    #   for post-create setup as its value.
    # * Each item of the script list is an object with a key 'script'
    #   referring to the path to the script and a key 'reboot'
    #   indicating whether reboot is required after running the
    #   script.
    #
    # Params:
    # * SSH destination, in the format like `user@ip_address`
    # * path to the tools directory
    # * cloud service: compshare, alicloud, etc.
    # * extra data for the cloud service
    local ssh_dest="${1:-}"
    local tools_dir="${2:-}"
    local cloud_service="${3:-}"
    local cloud_service_extra_data="${4:-}"

    [[ -z ${ssh_dest} \
      || -z ${cloud_service} \
      || -z ${tools_dir} ]] && return 1

    cloud_service="${cloud_service@L}"
    local config_yml="${tools_dir}/${cloud_service}/config.yml"

    echo 'Uploading tools to the VM ...' >&2
    scp -qr -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${tools_dir}" "${ssh_dest}":

    readarray -d '' post_create_scripts < \
        <(yq -0 '.scripts.post_create[].script' "${config_yml}")
    readarray -d '' reboot_required < \
        <(yq -0 '.scripts.post_create[].reboot' "${config_yml}")
    service_tools_dir="${tools_dir##*/}/${cloud_service}"
    for index in "${!post_create_scripts[@]}"; do
        script="${service_tools_dir}/${post_create_scripts[${index}]}"

        wait_for_vm_to_be_available "${ssh_dest}"
        echo "Running ${script} on the VM ..." >&2
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}" "\
                $(jq -r '[to_entries | .[] | "export \(.key)=\(.value | @sh);"] | join(" ")' \
                    <<< "${cloud_service_extra_data}") \
                bash ./${script}"

        if [[ ${reboot_required[${index}]} == true ]]; then
            echo 'Rebooting for setup to take effect ...' >&2
            ssh -t -o StrictHostKeyChecking=no \
                -o UserKnownHostsFile=/dev/null \
                "${ssh_dest}" "sudo reboot" || true
            wait_for_vm_to_be_available "${ssh_dest}"
        fi
    done
}