# syntax=docker/dockerfile:1

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Stage build order depending on the compute:
#     Compute Stage (CPU/GPU) -> Dependencies Stage -> Final Stage
#####################################################################
# Valid computes: cpu, gpu
ARG COMPUTE="cpu"


#####################################################################
# Compute Stage - CPU
# Choose an appropriate CPU compute image
#####################################################################
# * [buildpack-deps:24.04](https://github.com/docker-library/buildpack-deps/blob/master/ubuntu/noble/Dockerfile)
#   + [Created on 2025-11-14](https://hub.docker.com/layers/library/buildpack-deps/noble/images/sha256-40ee16e3f341497dfd8ae089abe47e60447f714389028726a8c21a75ac405be3)
FROM buildpack-deps:24.04@sha256:40ee16e3f341497dfd8ae089abe47e60447f714389028726a8c21a75ac405be3 AS cpu

# unzip and zip is required by SDKMAN!
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y zip unzip


#####################################################################
# Compute Stage - GPU
# Choose an appropriate GPU compute image
#####################################################################
# * [nvidia/cuda:12.8.1-devel-ubuntu24.04](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.8.1/ubuntu2404/devel/Dockerfile?ref_type=heads)
#   + [Created on 2025-03-14](https://hub.docker.com/layers/nvidia/cuda/12.8.1-devel-ubuntu24.04/images/sha256-4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43)
# * See also [AML GPU Base Image](https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04)
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43 AS gpu

# curl, unzip and zip is required by SDKMAN!
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y git zip unzip curl


#####################################################################
# Dependencies Stage
# Set up all dependencies.  This Stage is used by dev containers,
# because editable installation is required.
#####################################################################
FROM ${COMPUTE} AS deps

ARG COMPUTE

# Valid versions: 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

# Extra dependencies: dev, gpu, spark
ARG EXTRAS=""

ARG RECO_VENV_DIR=/root/.venvs/Recommenders

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root:root

# uv is in /root/.local/bin and is available after using bash -l
SHELL ["/bin/bash", "-lc"]

# Install yq
# See https://github.com/mikefarah/yq
RUN curl -LsSf https://github.com/mikefarah/yq/releases/download/v4.49.2/yq_linux_amd64.tar.gz | \
    tar xz && mv yq_linux_amd64 /usr/local/bin/yq

# Install SDKMAN!
RUN curl -LsSf "https://get.sdkman.io?ci=true" | bash

# Install OpenJDK
# sdk is available after source /root/.sdkman/bin/sdkman-init.sh
RUN if [[ "${EXTRAS}" =~ spark ]]; then \
      source /root/.sdkman/bin/sdkman-init.sh; \
      sdk install java 21.0.2-open; \
      sdk use java 21.0.2-open; \
    fi

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | bash

# Create virtual environment with uv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/opt/uv-cache/uv/python
ENV UV_CACHE_DIR=/opt/uv-cache/
RUN --mount=type=cache,target=/opt/uv-cache/uv/python \
    uv venv --python ${PYTHON_VERSION} ${RECO_VENV_DIR}


#####################################################################
# Final Stage
# Install Recommenders and its dependencies
#####################################################################
FROM deps AS final

# Git ref of Recommenders to install: main, staging, etc.
# Empty value ("") indicates editable installation of current clone
ARG GIT_REF="main"

ARG RECO_DIR="/root/Recommenders"

WORKDIR ${RECO_DIR}

# Copy Recommenders into the image
COPY ./ ${RECO_DIR}

# Install Recommenders
RUN --mount=type=cache,target=/opt/uv-cache/ \
    source ${RECO_VENV_DIR}/bin/activate && \
    if [ -z "${GIT_REF}" ]; then \
        uv pip install "recommenders${EXTRAS} @ ."; \
    else \
        uv pip install recommenders${EXTRAS} @ git+https://github.com/recommenders-team/recommenders@${GIT_REF}; \
    fi

# Setup Jupyter notebook
RUN source ${RECO_VENV_DIR}/bin/activate && \
    jupyter notebook --generate-config && \
    echo "c.MultiKernelManager.default_kernel_name = 'Recommenders'" >> /root/.jupyter/jupyter_notebook_config.py && \
    python -m ipykernel install --user --name Recommenders --display-name "Python (Recommenders)"

EXPOSE 8888
CMD source /root/.sdkman/bin/sdkman-init.sh && source ${RECO_VENV_DIR}/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.allow_origin='*' --IdentityProvider.token=''
