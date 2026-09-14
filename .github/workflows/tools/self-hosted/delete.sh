#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Delete the Docker containers and image on the self-hosted runner to
# free the space.
# 
# Params:
# * Name for the image tag and the code directory
#
# The following environment variables may need to be set:
# * SSH_DEST
#   + in the format like username@ip_address
######################################################################
set -euo pipefail
shopt -s inherit_errexit

unique_name="${1:-}"

[[ -z ${unique_name} ]] && exit 0

image_tag="${unique_name}"
recommenders_dir_name="${unique_name}"


#--------------------------------------------------------------------
# Remove the Docker containser and image
#--------------------------------------------------------------------
if [[ -z ${SSH_DEST:-} ]]; then
    if docker image ls | grep "${image_tag}" > /dev/null; then
        if docker container ls -a | grep "${image_tag}" > /dev/null; then
            container="$(docker ps -a | grep "${image_tag}" | awk '{print $1}')"
            docker container rm -f "${container}"
        fi
        docker image rm "${image_tag}"
    fi
else
    ssh -t -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=60 \
        -o ServerAliveCountMax=10 \
        "${SSH_DEST}" "\
            if docker image ls | grep '${image_tag}' > /dev/null; then \
                if docker container ls -a | grep '${image_tag}' > /dev/null; then \
                    container=\"\$(docker ps -a | grep '${image_tag}' | awk '{print \$1}')\"; \
                    docker container rm -f \"\${container}\"; \
                fi; \
                docker image rm '${image_tag}'; \
            fi; \
            rm -rf '${recommenders_dir_name}'"
fi
