#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

######################################################################
# Delete the Docker containers and image on the self-hosted runner to
# free the space.
# 
# Params:
# * Name for the image tag and the code directory
######################################################################
set -euo pipefail
shopt -s inherit_errexit

unique_name="${1:-}"

[[ -z ${unique_name} ]] && exit 0

image_tag="${unique_name}"


#--------------------------------------------------------------------
# Remove the Docker containser and image
#--------------------------------------------------------------------
if docker image ls | grep "${image_tag}" > /dev/null; then
    if docker container ls -a | grep "${image_tag}" > /dev/null; then
        container="$(docker ps -a | grep "${image_tag}" | awk '{print $1}')"
        docker container rm -f "${container}"
    fi
    docker image rm "${image_tag}"
fi
