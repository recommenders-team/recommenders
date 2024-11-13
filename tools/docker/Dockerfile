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
# Dependencies Stage
# Set up all dependencies.  This Stage is used by dev containers,
# because editable installation is required.
#####################################################################
FROM ${COMPUTE} AS deps

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
ARG CONDA_PREFIX="/root/conda"
ARG CONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"

# Install Conda
RUN wget -qO ${CONDA_FILE} ${CONDA_URL} && \
    bash ${CONDA_FILE} -bf -p ${CONDA_PREFIX} && \
    ${CONDA_PREFIX}/bin/conda clean -ay && \
    rm -rf ${CONDA_PREFIX}/pkgs && \
    rm ${CONDA_FILE} && \
    ${CONDA_PREFIX}/bin/conda init bash && \
    ${CONDA_PREFIX}/bin/conda config --set auto_activate_base false 

# Create Conda environment
RUN ${CONDA_PREFIX}/bin/conda create -n Recommenders -c conda-forge -y \
        python=${PYTHON_VERSION} \
        pip \
        $(if [[ "${EXTRAS}" =~ spark ]]; then echo "openjdk=${JDK_VERSION}"; else echo ""; fi)


#####################################################################
# Final Stage
# Install Recommenders
#####################################################################
FROM deps AS final

ARG RECO_LOCAL_DIR="./"
ARG RECO_DIR="/root/Recommenders"
ARG RECO_GIT_URL="git+https://github.com/recommenders-team/recommenders.git"

# Copy Recommenders into the image
COPY ${RECO_LOCAL_DIR} ${RECO_DIR}

# Install Recommenders and its dependencies
RUN source ${CONDA_PREFIX}/bin/activate && \
    conda activate Recommenders && \
    if [ -z "${GIT_REF}" ]; then \
        pip install ${RECO_DIR}${EXTRAS}; \
    else \
        pip install recommenders${EXTRAS}@${RECO_GIT_URL}@${GIT_REF}; \
    fi && \
    jupyter notebook --generate-config && \
    echo "c.MultiKernelManager.default_kernel_name = 'Recommenders'" >> /root/.jupyter/jupyter_notebook_config.py && \
    python -m ipykernel install --user --name Recommenders --display-name "Python (Recommenders)"

# Activate Recommenders Conda environment
ENV CONDA_SHLVL='1'
ENV CONDA_PROMPT_MODIFIER='(Recommenders) '
ENV JAVA_HOME="${CONDA_PREFIX}/lib/jvm"
ENV JAVA_LD_LIBRARY_PATH="${JAVA_HOME}/lib/server"
ENV PATH="${CONDA_PREFIX}/envs/Recommenders/bin:${CONDA_PREFIX}/condabin:${PATH}"
ENV CONDA_PREFIX="${CONDA_PREFIX}/envs/Recommenders"
ENV PS1='(Recommenders) \[\]\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\u@\h:\w\$ \[\]'

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.allow_origin='*'", "--IdentityProvider.token=''"]