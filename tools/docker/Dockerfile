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
#   + [Created on 2026-04-19](https://hub.docker.com/layers/library/buildpack-deps/24.04/images/sha256-6ddc88f2b2d1972421e7ed175f105b4d18785751fd896edd7f435922852e9673)
FROM buildpack-deps:24.04@sha256:6ddc88f2b2d1972421e7ed175f105b4d18785751fd896edd7f435922852e9673 AS cpu


#####################################################################
# Compute Stage - GPU
# Choose an appropriate GPU compute image
#####################################################################
# * [nvidia/cuda:13.2.1-devel-ubuntu24.04](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/13.2.1/ubuntu2404/devel/Dockerfile)
#   + [Created on 2026-04-20](https://hub.docker.com/layers/nvidia/cuda/13.2.1-devel-ubuntu24.04/images/sha256-0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf)
# * See also [AML GPU Base Image](https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04)
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf AS gpu


#####################################################################
# Dependencies Stage
# Set up all dependencies.  This Stage is used by dev containers,
# because editable installation is required.
#####################################################################
FROM ${COMPUTE} AS deps

ARG COMPUTE

# Extra dependencies: dev, gpu, spark
ARG EXTRAS=""

# Valid versions: 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

ARG RECO_VENV_DIR=/root/.venvs/Recommenders

# HTTPS proxy certificate if needed
ARG VM_PROXY_CERTIFICATE=''

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root:root

# uv is in /root/.local/bin and is available after using bash -l
SHELL ["/bin/bash", "-lc"]

# Add HTTPS proxy CA certificate
# * ca-certificates is required and installed in all base images
RUN if [[ -n "${VM_PROXY_CERTIFICATE}" ]]; then \
        echo "${VM_PROXY_CERTIFICATE}" > /usr/local/share/ca-certificates/vm_proxy_cert.crt \
        && update-ca-certificates; \
    fi

# curl, unzip and zip is required by SDKMAN!
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y curl git unzip zip

# Install SDKMAN!
RUN curl -LsSf --retry 5 --retry-delay 10 --retry-all-errors \
        -o sdkman.sh \
        "https://get.sdkman.io?ci=true" \
    && bash sdkman.sh \
    && rm sdkman.sh

# Install OpenJDK
# sdk is available after source /root/.sdkman/bin/sdkman-init.sh
RUN if [[ "${EXTRAS}" =~ spark ]]; then \
      source /root/.sdkman/bin/sdkman-init.sh \
      && sdk install java 21.0.2-open \
      && sdk use java 21.0.2-open; \
    fi

# Install uv
RUN curl -LsSf --retry 5 --retry-delay 10 --retry-all-errors \
        -o uv_install.sh \
        https://astral.sh/uv/install.sh \
    && bash uv_install.sh \
    && rm uv_install.sh

# Create virtual environment with uv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_CACHE_DIR=/opt/uv-cache/uv/python
ENV UV_CACHE_DIR=/opt/uv-cache/
ENV UV_HTTP_TIMEOUT=300
ARG UV_INSECURE_HOST
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

# Use PyPI as the default pip index,
# otherwise, use the specified.
ARG VM_PIP_INDEX_URL=''

ARG RECO_DIR="/root/Recommenders"

WORKDIR ${RECO_DIR}

# Copy Recommenders into the image
COPY ./ ${RECO_DIR}

# Install Recommenders
RUN --mount=type=cache,target=/opt/uv-cache/ \
    source ${RECO_VENV_DIR}/bin/activate && \
    { count=0; \
      while true; do \
          if [ -z "${GIT_REF}" ]; then \
              uv pip install --index "${VM_PIP_INDEX_URL}" "recommenders${EXTRAS} @ ."; \
          else \
              uv pip install --index "${VM_PIP_INDEX_URL}" recommenders${EXTRAS} @ git+https://github.com/recommenders-team/recommenders@${GIT_REF}; \
          fi \
          && break \
          || { ((count++)); if [[ $count -lt 5 ]]; then sleep 5; else exit 1; fi; } \
      done; \
    }

# Setup Jupyter notebook
RUN source ${RECO_VENV_DIR}/bin/activate && \
    jupyter notebook --generate-config && \
    echo "c.MultiKernelManager.default_kernel_name = 'Recommenders'" >> /root/.jupyter/jupyter_notebook_config.py && \
    python -m ipykernel install --user --name Recommenders --display-name "Python (Recommenders)"

EXPOSE 8888
CMD source /root/.sdkman/bin/sdkman-init.sh && source ${RECO_VENV_DIR}/bin/activate && jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --ServerApp.allow_origin='*' --IdentityProvider.token=''
