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

# Valid versions: 3.8, 3.9, 3.10, 3.11
ARG PYTHON_VERSION="3.11"

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /root
USER root:root

SHELL ["/bin/bash", "-c"]

# Install Conda
RUN wget -qO /tmp/conda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" && \
    bash /tmp/conda.sh -bf -p /root/conda && \
    /root/conda/bin/conda clean -ay && \
    rm -rf /root/conda/pkgs && \
    rm /tmp/conda.sh && \
    /root/conda/bin/conda init bash && \
    /root/conda/bin/conda config --set auto_activate_base false 

# Create Conda environment
RUN /root/conda/bin/conda create -n Recommenders -c conda-forge -y python=${PYTHON_VERSION} pip


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

ARG JDK_VERSION="21"

ARG RECO_DIR="/root/Recommenders"

# Copy Recommenders into the image
COPY ./ ${RECO_DIR}

# Install Recommenders and its dependencies
RUN source /root/conda/bin/activate && \
    conda activate Recommenders && \
    if [[ "${EXTRAS}" =~ spark ]]; then conda install -c conda-forge -y "openjdk=${JDK_VERSION}"; fi && \
    if [ -z "${GIT_REF}" ]; then \
        pip install ${RECO_DIR}${EXTRAS}; \
    else \
        pip install recommenders${EXTRAS}@git+https://github.com/recommenders-team/recommenders.git@${GIT_REF}; \
    fi && \
    jupyter notebook --generate-config && \
    echo "c.MultiKernelManager.default_kernel_name = 'Recommenders'" >> /root/.jupyter/jupyter_notebook_config.py && \
    python -m ipykernel install --user --name Recommenders --display-name "Python (Recommenders)"

# Activate Recommenders Conda environment
ENV PS1='(Recommenders) \[\]\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\u@\h:\w\$ \[\]'
ENV PATH="/root/conda/envs/Recommenders/bin:/root/conda/condabin:${PATH}"
ENV CONDA_SHLVL='1'
ENV CONDA_PROMPT_MODIFIER='(Recommenders) '
ENV CONDA_PREFIX="/root/conda/envs/Recommenders"
ENV CONDA_EXE="/root/conda/bin/conda"
ENV CONDA_PYTHON_EXE="/root/conda/bin/python"
ENV JAVA_HOME="/root/conda/envs/Recommenders/lib/jvm"
ENV JAVA_LD_LIBRARY_PATH="${JAVA_HOME}/lib/server"

# Setup Jupyter notebook
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.allow_origin='*'", "--IdentityProvider.token=''"]
