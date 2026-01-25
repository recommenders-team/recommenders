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


#####################################################################
# Compute Stage - GPU
# Choose an appropriate GPU compute image
#####################################################################
# * [nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/13.1.1/ubuntu2404/devel/cudnn/Dockerfile)
#   + [Created on 2026-01-23](https://hub.docker.com/layers/nvidia/cuda/13.1.1-cudnn-devel-ubuntu24.04/images/sha256-8be896813bf8bf805a6eca85969bf0b2957cba9e3474e7759f56fe9206d838b8)
# * See also [AML GPU Base Image](https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04)
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04@sha256:8be896813bf8bf805a6eca85969bf0b2957cba9e3474e7759f56fe9206d838b8 AS gpu

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y wget git zip unzip curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


#####################################################################
# Dependencies Stage
# Set up all dependencies.  This Stage is used by dev containers,
# because editable installation is required.
#####################################################################
FROM ${COMPUTE} AS deps

# Valid versions: 3.8, 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root:root

SHELL ["/bin/bash", "-lc"]

# Install yq
# See https://github.com/mikefarah/yq
RUN wget -qO- https://github.com/mikefarah/yq/releases/download/v4.49.2/yq_linux_amd64.tar.gz | \
    tar xz && mv yq_linux_amd64 /usr/local/bin/yq

# Install SDKMAN!
RUN wget -qO- "https://get.sdkman.io?ci=true&rcupdate=false" | bash

# Install uv
RUN wget -qO- https://astral.sh/uv/install.sh | bash


#####################################################################
# Final Stage
# Install Recommenders
#####################################################################
FROM deps AS final

# Extra dependencies: dev, gpu, spark
ARG EXTRAS=""

# Git ref of Recommenders to install: main, staging, etc.
# Empty value ("") indicates editable installation of current clone
ARG GIT_REF="main"

WORKDIR /root/Recommenders

# Install Recommenders and its dependencies
RUN if [[ "${EXTRAS}" =~ spark ]]; then \
      sdk install java 21.0.2-open; \
      sdk use java 21.0.2-open; \
    fi

# Copy Recommenders into the image
COPY ./ ./

RUN uv venv --python ${PYTHON_VERSION}
RUN source .venv/bin/activate && \
    if [ -z "${GIT_REF}" ]; then \
        uv pip install "recommenders${EXTRAS} @ ."; \
    else \
        uv pip install recommenders${EXTRAS} @ git+https://github.com/recommenders-team/recommenders@${GIT_REF}; \
    fi && \
    jupyter notebook --generate-config && \
    echo "c.MultiKernelManager.default_kernel_name = 'Recommenders'" >> /root/.jupyter/jupyter_notebook_config.py && \
    python -m ipykernel install --user --name Recommenders --display-name "Python (Recommenders)"

RUN source .venv/bin/activate && \
    if [[ ! ${COMPUTE} =~ gpu ]]; then \
      uv pip install pytest-xdist; \
    fi

# Setup Jupyter notebook
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.allow_origin='*'", "--IdentityProvider.token=''"]
