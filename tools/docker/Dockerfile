# syntax=docker/dockerfile:1

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

#####################################################################
# Stage build order depending on the compute, extras, python version
# and virtual environment:
#        Compute Stage (CPU/GPU)
#     -> Base Stage
#     -> Virtual Environment Stage (Conda/Venv/Virtualenv)
#     -> Dependencies Stage
#     -> Final Stage
#####################################################################
# Valid computes: cpu, gpu
ARG COMPUTE="cpu"


#####################################################################
# Compute Stage - CPU
# Choose an appropriate CPU compute image
#####################################################################
# * [buildpack-deps:24.04](https://github.com/docker-library/buildpack-deps/blob/master/ubuntu/noble/Dockerfile)
#   + [Created on 2024-08-17](https://hub.docker.com/layers/library/buildpack-deps/noble/images/sha256-dbfee7e7ee2340b0d6567efd3a8a9281ce45ee78598485b4d7a7f09fe641811a)
FROM buildpack-deps@sha256:dbfee7e7ee2340b0d6567efd3a8a9281ce45ee78598485b4d7a7f09fe641811a AS cpu


#####################################################################
# Compute Stage - GPU
# Choose an appropriate GPU compute image
#####################################################################
# * [nvidia/cuda:12.6.1-devel-ubuntu24.04](https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.6.1/ubuntu2404/devel/Dockerfile)
#   + [Created on 2024-09-13](https://hub.docker.com/layers/nvidia/cuda/12.6.1-devel-ubuntu24.04/images/sha256-bfc293f21611f3c47a3442cf6516ebfe99d529926a4bef4bc389ef02fd038800)
# * See also [AML GPU Base Image](https://github.com/Azure/AzureML-Containers/blob/master/base/gpu/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04)
FROM nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu24.04@sha256:bfc293f21611f3c47a3442cf6516ebfe99d529926a4bef4bc389ef02fd038800 AS gpu

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y wget git && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


#####################################################################
# Base Stage
# Basic settings for all stages built on top of compute stage
#####################################################################
FROM ${COMPUTE}

# Extra dependencies: dev, gpu, spark
ARG EXTRAS=""

# Git ref of Recommenders to install: main, staging, etc.
# Empty value ("") indicates editable installation of current clone
ARG GIT_REF="main"

ARG JDK_VERSION="21"

# Valid versions: 3.8, 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /root
USER root:root

SHELL ["/bin/bash", "-c"]

ARG CONDA_FILE="/tmp/conda.sh"
ARG CONDA_PREFIX="conda"
ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG ENV_NAME="Recommenders"
ARG RECO_LOCAL_DIR="./"
ARG RECO_DIR="/tmp/recommenders"

# Install Conda
RUN wget -qO ${CONDA_FILE} ${CONDA_URL} && \
    bash ${CONDA_FILE} -bf -p ${CONDA_PREFIX} && \
    ${CONDA_PREFIX}/bin/conda clean -ay && \
    rm -rf ${CONDA_PREFIX}/pkgs && \
    rm ${CONDA_FILE} && \
    ${CONDA_PREFIX}/bin/conda init bash && \
    ${CONDA_PREFIX}/bin/conda config --set auto_activate_base false 

# Create Conda environment
RUN JDK=$(if [[ "${EXTRAS}" =~ spark ]]; then echo "openjdk=${JDK_VERSION}"; else echo ""; fi) \
    ${CONDA_PREFIX}/bin/conda create -n ${ENV_NAME} -c conda-forge -y python=${PYTHON_VERSION} pip ${JDK}; \
    echo "conda activate ${ENV_NAME}" >> ~/.bashrc

# Copy Recommenders into the image
COPY ${RECO_LOCAL_DIR} ${RECO_DIR}

# Install Recommenders and its dependencies
RUN RECO=$(if [ -z "${GIT_REF}" ]; then echo "${RECO_DIR}"; else echo "recommenders"; fi); \
    RECO_URL=$(if [ -z "${GIT_REF}" ]; then echo ""; else echo "@git+https://github.com/recommenders-team/recommenders.git@${GIT_REF}"; fi); \
    source ${CONDA_PREFIX}/bin/activate && \
    conda activate ${ENV_NAME} && \
    pip install ${RECO}$(if [ -z "${EXTRAS}" ]; then echo ""; else echo "[${EXTRAS}]"; fi)${RECO_URL} && \
    python -m ipykernel install --user --name ${ENV_NAME} --display-name "Python (${ENV_NAME})"

# Source ~/.bashrc to activate the Conda environment
SHELL ["/bin/bash", "-i", "-c"]
ENV ENV="/root/.bashrc"
ENV BASH_ENV="${ENV}"
