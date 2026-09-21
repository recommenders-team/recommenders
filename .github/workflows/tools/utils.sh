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

apply_tf_config() {
    # Apply the Terraform configuration to create a VM with possible
    # input variable values in the specified directory.
    #
    # NOTE: 
    # This function assumes there is a input variable named
    # 'unique_name' representing the name of the VM created.
    #
    # Params:
    # * Name for the resources such as VM
    # * the directory containing the Terraform configuration
    # * a JSON object of input variables each with possible values
    #   + For example,
    #
    #     {
    #         "instance_type_family": [
    #             "ecs.gn8is",
    #             "ecs.gn7i",
    #             "ecs.gn6i"
    #         ],
    #         "region": [
    #             "ap-southeast-5",
    #             "ap-northeast-1",
    #             "eu-central-1",
    #             "ap-southeast-1"
    #         ]
    #     }
    local unique_name="${1:-}"
    local tf_config_dir="${2:-./}"
    local input_vars="${3:-}"

    [[ -z ${unique_name} ]] \
        && echo 'No name specified!' >&2 \
        && return 1

    local var_combinations
    readarray -t var_combinations < \
        <(get_input_var_combinations "${input_vars}")

    local index
    for index in "${!var_combinations[@]}"; do
        local combination="${var_combinations[${index}]}"
        echo "${combination}" > "${tf_config_dir}/reco.auto.tfvars.json"

        echo "* Trying with ${combination} ..."
        terraform -chdir="${tf_config_dir}" apply -auto-approve \
            -var "unique_name=${unique_name}" && return

        echo '* Cleaning up ...'
        terraform -chdir="${tf_config_dir}" destroy -auto-approve \
            -var "unique_name=${unique_name}"
    done
    echo 'All required resources are sold out!' && return 1
}

get_env_exports() {
    # Generate shell environment variable export statements for
    # key-value pairs in $1
    #
    # For example, if $1 is
    #
    #     {
    #        "VM_DOCKER_MIRROR_URL": "https://docker.sparkcr.cn",
    #        "VM_HTTP_PROXY": "http://172.168.2.6:3141"
    #     }
    #
    # Then it returns
    #     export VM_DOCKER_MIRROR_URL=https://docker.sparkcr.cn;
    #     export VM_HTTP_PROXY=http://172.168.2.6:3141";
    #
    # Params:
    # * JSON string
    local json_data="${1:-}"

    local env_exports
    env_exports="$(jq -r '
        [to_entries | .[] | "export \(.key)=\(.value | @sh);"]
        | join(" ")' \
        <<< "${json_data}")"
    echo "${env_exports}"
}

get_input_var_combinations() {
    # Return all combinations of input variables from $1.
    #
    # Params:
    # * a JSON object of input variables each with possible values
    #   + For example,
    #
    #     {
    #         "instance_type_family": [
    #             "ecs.gn8is",
    #             "ecs.gn7i",
    #             "ecs.gn6i"
    #         ],
    #         "region": [
    #             "ap-southeast-5",
    #             "ap-northeast-1",
    #             "eu-central-1",
    #             "ap-southeast-1"
    #         ]
    #     }
    #
    #     and it returns:
    #   
    #     {"instance_type_family":"ecs.gn8is","region":"ap-southeast-5"}
    #     {"instance_type_family":"ecs.gn8is","region":"ap-northeast-1"}
    #     {"instance_type_family":"ecs.gn8is","region":"eu-central-1"}
    #     {"instance_type_family":"ecs.gn8is","region":"ap-southeast-1"}
    #     {"instance_type_family":"ecs.gn7i","region":"ap-southeast-5"}
    #     {"instance_type_family":"ecs.gn7i","region":"ap-northeast-1"}
    #     {"instance_type_family":"ecs.gn7i","region":"eu-central-1"}
    #     {"instance_type_family":"ecs.gn7i","region":"ap-southeast-1"}
    #     {"instance_type_family":"ecs.gn6i","region":"ap-southeast-5"}
    #     {"instance_type_family":"ecs.gn6i","region":"ap-northeast-1"}
    #     {"instance_type_family":"ecs.gn6i","region":"eu-central-1"}
    #     {"instance_type_family":"ecs.gn6i","region":"ap-southeast-1"}
    local input_vars="${1:-}"
    local combinations
    combinations="$(jq -c '
        to_entries
        | reduce .[] as $cur_var (
            [{}];
            [
                .[] as $combination
                | $cur_var.value[] as $cur_var_val
                | $combination + {($cur_var.key): $cur_var_val}
            ])
        | .[]' <<< "${input_vars}")" \
        || { echo 'Incorrect input variables!' >&2; exit 1; }
    echo "${combinations}"
}

pre_image_build() {
    # Prepare for the Docker image build.
    #
    # DNS resolution of URLs pointing to and downloading from GitHub
    # may fail on the VM side, so this function copies those files
    # from GitHub on the Actions runner side, including:
    # * SDKMan installation zip files
    # * Recommenders repo
    #
    # Params:
    # * SSH destination of the VM in the form of username@ip_address
    # * path to the Dockerfile
    # * path to config.yml
    # * name of the directory containing the recommenders repo files
    local ssh_dest="${1:-}"
    local dockerfile="${2:-}"
    local config_yml="${3:-}"
    local recommenders_dir_name="${4:-}"

    [[ -z ${ssh_dest} \
      || -z ${dockerfile} \
      || -z ${recommenders_dir_name} ]] \
      && echo 'Parameter error!' >&2 return 1

    echo 'Preparing for the Docker image build ...'
    echo '* Dowloading SDKMan files ...'
    local sdkman_sh='sdkman.sh'
    local sdkman_zip='sdkman.zip'
    local sdkman_native_zip='sdkman-native.zip'
    curl -LsSf -o "${sdkman_sh}" https://get.sdkman.io/?ci=true
    local sdkman_service
    sdkman_service="$(grep 'SDKMAN_SERVICE=' "${sdkman_sh}" \
        | cut -d '"' -f 2)"
    local sdkman_version
    sdkman_version="$(grep 'SDKMAN_VERSION=' "${sdkman_sh}" \
        | cut -d '"' -f 2)"
    local sdkman_native_version
    sdkman_native_version="$(grep 'SDKMAN_NATIVE_VERSION=' \
        "${sdkman_sh}" | cut -d '"' -f 2)"
    curl -LsSf -o "${sdkman_zip}" "${sdkman_service}/broker/download/sdkman/install/${sdkman_version}/linuxx64"
    curl -LsSf -o "${sdkman_native_zip}" "${sdkman_service}/broker/download/native/install/${sdkman_native_version}/linuxx64"

    echo '* Configuring SDKMan in Dockerfile ...'
    sed -i \
        -e "/USER/a \
            WORKDIR /root \
            COPY ./${sdkman_zip} /root/${sdkman_zip} \
            COPY ./${sdkman_native_zip} /root/${sdkman_native_zip}" \
        -e  "/get.sdkman.io/a \
            && sed -i -e \"/* Downloading/a if [[ ! -f /root/${sdkman_zip} && ! -f /root/${sdkman_native_zip} ]]; then\" \\\\\\
                    -e \"/download\\\/sdkman/a else cp /root/${sdkman_zip} \\\\\"\\\\$\{sdkman_zip_file\}\\\\\"; fi\" \\\\\\
                    -e \"/download\\\/native/a else cp /root/${sdkman_native_zip} \\\\\"\\\\$\{sdkman_zip_file\}\\\\\"; fi\" \\\\\\
                    ${sdkman_sh} \\\\" \
        "${dockerfile}"

    echo '* Configuring APT in Dockerfile ...'
    if [[ -f ${config_yml} ]]; then
        apt_mirror="$(yq -o json "${config_yml}" \
            | jq -r '.apt_mirror // empty')"
        if [[ -n ${apt_mirror} ]]; then
            sed -i "/SHELL /a \
                RUN sed -i -e \"s#archive.ubuntu.com#${apt_mirror}#g\" \\\\\\
                        -e \"s#security.ubuntu.com#${apt_mirror}#g\" \\\\\\
                        /etc/apt/sources.list.d/ubuntu.sources" \
                "${dockerfile}"
        fi
    fi

    echo '* Uploading recommenders ...'
    local recomenders_tar="${recommenders_dir_name}.tar"
    tar cf "../${recomenders_tar}" ./*
    scp -q -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "../${recomenders_tar}" "${ssh_dest}":
    rm -rf "../${recomenders_tar}"
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "${ssh_dest}" "\
            mkdir ${recommenders_dir_name} \
            && tar xf ${recomenders_tar} -C ${recommenders_dir_name} \
            && rm -rf ${recomenders_tar}"
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
    # * SSH private key or file containing the base64-encoded login
    #   password 
    local ssh_dest="${1:-}"
    local sshkey_or_passfile="${2:-}"
    [[ -z ${ssh_dest} \
      || -z ${sshkey_or_passfile} \
      || ! -f ${sshkey_or_passfile} ]] \
      && echo 'Parameter error!' >&2 return 1

    echo 'Setting up SSH key for login ...'
    local ssh_key_type
    if ssh_key_type="$(ssh-keygen -l -f "${sshkey_or_passfile}" \
        2>/dev/null)"; then
        local ssh_key="${sshkey_or_passfile}"
        ssh_key_type="$(echo "${ssh_key_type}" \
            | cut -d '(' -f 2 \
            | cut -d ')' -f 1)"
        mv -f "${ssh_key}" "${HOME}/.ssh/id_${ssh_key_type@L}"
    else
        local encoded_password_file="${sshkey_or_passfile}"
        local key_file="${HOME}/.ssh/id_ed25519"
        local sshd_config="/etc/ssh/sshd_config"

        echo '* Generating SSH key ...'
        if [[ ! -f ${key_file} || ! -f ${key_file}.pub ]]; then
            ssh-keygen -q -t ed25519 -N '' -f "${key_file}"
        fi

        echo '* Deplying SSH key ...'
        local -x SSHPASS
        read -r SSHPASS < <(cat "${encoded_password_file}" \
            | tr -d '\n' | base64 -d) || true
        rm -rf "${encoded_password_file}"
        run_cmd_retry sshpass -e ssh-copy-id \
            -i "${key_file}.pub" \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}"

        echo '* Disabling SSH password authentication ...'
        ssh -t -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            "${ssh_dest}" "\
                sudo sed -i -E 's/^[[:space:]#]*PasswordAuthentication.*/PasswordAuthentication no/' ${sshd_config}; \
                sudo systemctl reload ssh"
    fi
}

snap_install_retry() {
    # Run snap install "$@" and retry "$1" times
    # (5 by default) on failure.
    #
    # Params:
    # * (optional) number of attempts
    local num_attempts=5
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        num_attempts="$1"
        shift
    fi

    run_cmd_retry "${num_attempts}" sudo snap install "$@"
}

update_json() {
    # Update a JSON with another JSON
    #
    # Params:
    # * the original JSON
    # * the JSON with all updates
    local original="${1:-}"
    local updates="${2:-}"
    [[ -z ${updates} || -z ${original} ]] \
      && echo 'Parameter error!' >&2 && return 1

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
        echo 'Waiting for processes releasing /var/lib/apt/lists/lock ...'
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
    [[ -z ${ssh_dest} ]] && echo 'Parameter error!' >&2 && return 1

    echo 'Waiting for the VM to be available ...'
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
        # Set timeout to (5 + 5) * 60 = 600 seconds
        [[ "${count}" -gt 60 ]] && echo 'Time out!' >&2 && return 1
        count=$((count + 1))
        echo '* Still waiting ...'
        sleep 5
    done
}
