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
FROM ${COMPUTE} AS base

# Extra dependencies: dev, gpu, spark
ARG EXTRAS=""

# Git ref of Recommenders to install: main, staging, etc.
# Empty value ("") indicates editable installation of current clone
ARG GIT_REF="main"

# Valid versions: 3.8, 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

# Valid virtual environments: venv, virtualenv, conda
ARG VIRTUAL_ENV="conda"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

WORKDIR /root
USER root:root

# -i option used to source bashrc (thus activating conda env) at every RUN statement
SHELL ["/bin/bash", "-i", "-c"]

# Exit if VIRTUAL_ENV is not specified correctly
RUN if ! [[ "${VIRTUAL_ENV}" =~ ^(conda|venv|virtualenv)$ ]]; then \
        echo "VIRTUAL_ENV is now '${VIRTUAL_ENV}', which should be either 'conda', 'venv' or 'virtualenv'"; \
        exit 1; \
    fi

# Install system Python
RUN apt-get update && \
    apt-get install -y software-properties-common  # Required for add-apt-repository
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python${PYTHON_VERSION}-dev


#####################################################################
# Virtual Environment Stage - Conda
#####################################################################
FROM base AS conda

ARG CONDA_FILE="/tmp/conda.sh"
ARG CONDA_PREFIX="conda"
ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
ARG ENV_HOME="${CONDA_PREFIX}/envs/Recommenders"

# Install Conda
RUN if [ "${VIRTUAL_ENV}" = "conda" ] ; then \
        wget -qO ${CONDA_FILE} ${CONDA_URL} && \
        bash ${CONDA_FILE} -bf -p ${CONDA_PREFIX} && \
        ${CONDA_PREFIX}/bin/conda clean -ay && \
        rm -rf ${CONDA_PREFIX}/pkgs && \
        rm ${CONDA_FILE} && \
        ${CONDA_PREFIX}/bin/conda init bash && \
        ${CONDA_PREFIX}/bin/conda config --set auto_activate_base false; \ 
    fi


#####################################################################
# Virtual Environment Stage - Venv
#####################################################################
FROM base AS venv

ARG ENV_HOME=".venv"

# Install venv
RUN apt-get install -y python${PYTHON_VERSION}-venv

# Create virtual environment
RUN python${PYTHON_VERSION} -m venv ${ENV_HOME}


#####################################################################
# Virtual Environment Stage - Virtualenv
#####################################################################
FROM base AS virtualenv

ARG ENV_HOME=".virtualenv"

# Install virtualenv
RUN apt-get install -y python3-virtualenv

# Create virtual environment
RUN virtualenv -p python${PYTHON_VERSION} ${ENV_HOME}


#####################################################################
# Dependencies Stage
#####################################################################
FROM ${VIRTUAL_ENV} AS deps

ARG JDK_VERSION="21"


RUN echo "ENV_HOME: ${ENV_HOME}"
RUN if [[ "${VIRTUAL_ENV}" =~ ^(venv|virtualenv)$ ]]; then \
        if [[ "${EXTRAS}" =~ spark ]]; then \
            apt-get install -y openjdk-${JDK_VERSION}-jdk; \
        fi; \
        source ${ENV_HOME}/bin/activate; \
        pip install --upgrade pip setuptools; \
        echo "source ${ENV_HOME}/bin/activate" >> ~/.bashrc; \
    fi
RUN if [[ "${VIRTUAL_ENV}" =~ ^(conda)$ ]]; then \
        JDK=$(if [[ "${EXTRAS}" =~ spark ]]; then echo "openjdk=${JDK_VERSION}"; else echo ""; fi) \
        conda create -n ${ENV_HOME##*/} -c conda-forge -y python=${PYTHON_VERSION} pip ${JDK}; \
        echo "conda activate ${ENV_HOME##*/}" >> ~/.bashrc; \
    fi


#####################################################################
# Final Stage
#####################################################################
FROM deps AS final

ARG RECO_LOCAL_DIR="./"
ARG RECO_DIR="/tmp/recommenders"

COPY ${RECO_LOCAL_DIR} ${RECO_DIR}

RUN RECO=$(if [ -z "${GIT_REF}" ]; then echo "${RECO_DIR}"; else echo "recommenders"; fi); \
    RECO_URL=$(if [ -z "${GIT_REF}" ]; then echo ""; else echo "@git+https://github.com/recommenders-team/recommenders.git@${GIT_REF}"; fi); \
    pip install ${RECO}$(if [ -z "${EXTRAS}" ]; then echo ""; else echo "[${EXTRAS}]"; fi)${RECO_URL} && \
    python -m ipykernel install --user --name ${VIRTUAL_ENV} --display-name "Python (${VIRTUAL_ENV})"
