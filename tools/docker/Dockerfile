# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

ARG ENV="cpu"
ARG HOME="/root"

FROM ubuntu:18.04 AS base

LABEL maintainer="Microsoft Recommender Project <RecoDevTeam@service.microsoft.com>"

ARG HOME
ENV HOME="${HOME}"
WORKDIR ${HOME}

# Install base dependencies, cmake (for xlearn) and libpython3.6 (for cornac)
RUN apt-get update && \
    apt-get install -y curl build-essential cmake libpython3.6

# Install Anaconda
ARG ANACONDA="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
RUN curl ${ANACONDA} -o anaconda.sh && \
    /bin/bash anaconda.sh -b -p conda && \
    rm anaconda.sh && \
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH="${HOME}/conda/bin:${PATH}"

# Python version supported by recommenders
RUN conda install python=3.6


###########
# CPU Stage
###########
FROM base AS cpu

RUN pip install recommenders[xlearn,examples]


###############
# PySpark Stage
###############
FROM base AS pyspark

# Install Java version 8
RUN apt-get update && \
    apt-get install -y libgomp1 openjdk-8-jre

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64" \
    PYSPARK_PYTHON="${HOME}/conda/bin/python" \
    PYSPARK_DRIVER_PYTHON="${HOME}/conda/bin/python"

# Install dependencies in Conda environment
RUN pip install recommenders[spark,xlearn,examples]


###########
# GPU Stage
###########
FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04 AS gpu

ARG HOME
WORKDIR ${HOME}

# Get up to date with base
COPY --from=base ${HOME} .
RUN apt-get update && \
    apt-get install -y build-essential cmake libpython3.6
ENV PATH="${HOME}/conda/bin:${PATH}"

# Install dependencies in Conda environment
RUN pip install recommenders[gpu,xlearn,examples] -f https://download.pytorch.org/whl/cu100/torch_stable.html


############
# Full Stage
############
FROM gpu AS full

ARG HOME
WORKDIR ${HOME}

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64" \
    PYSPARK_PYTHON="${HOME}/conda/bin/python" \
    PYSPARK_DRIVER_PYTHON="${HOME}/conda/bin/python"

# Install dependencies in Conda environment
RUN pip install recommenders[all,examples] -f https://download.pytorch.org/whl/cu100/torch_stable.html

#############
# Final Stage
#############
FROM $ENV AS final

# Setup Jupyter notebook configuration
ENV NOTEBOOK_CONFIG="${HOME}/.jupyter/jupyter_notebook_config.py"
RUN mkdir ${HOME}/.jupyter && \
    echo "c.NotebookApp.token = ''" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.allow_root = True" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >> ${NOTEBOOK_CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >> ${NOTEBOOK_CONFIG}

# Register the environment with Jupyter
RUN python -m ipykernel install --user --name base --display-name "Python (base)"

ARG HOME
WORKDIR ${HOME}

EXPOSE 8888
CMD ["jupyter", "notebook"]
