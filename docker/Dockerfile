# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

ARG ENV="cpu"
ARG HOME="/root"

FROM ubuntu:18.04 AS base

LABEL maintainer="Microsoft Recommender Project <RecoDevTeam@service.microsoft.com>"

ARG HOME
ENV HOME="${HOME}"
WORKDIR ${HOME}

# Install base dependencies
RUN apt-get update && \
    apt-get install -y curl git \
    apt-get install cmake

# Install Anaconda
ARG ANACONDA="https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh"
RUN curl ${ANACONDA} -o anaconda.sh && \
    /bin/bash anaconda.sh -b -p conda && \
    rm anaconda.sh && \
    echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
ENV PATH="${HOME}/conda/bin:${PATH}"

# Clone Recommenders repo
ARG BRANCH="master"
RUN git clone --depth 1 --single-branch -b ${BRANCH} https://github.com/microsoft/recommenders

# Setup Jupyter notebook configuration
ENV NOTEBOOK_CONFIG="${HOME}/.jupyter/jupyter_notebook_config.py"
RUN mkdir ${HOME}/.jupyter && \
    echo "c.NotebookApp.token = ''" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.allow_root = True" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >> ${NOTEBOOK_CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >> ${NOTEBOOK_CONFIG}


# CPU Stage
FROM base AS cpu

# Setup Conda environment
RUN python recommenders/scripts/generate_conda_file.py --name base


# PySpark Stage
FROM base AS pyspark

# Install base dependencies
RUN apt-get update && \
    apt-get install -y libgomp1 openjdk-8-jre

# Install Spark
ARG SPARK="http://archive.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz"
RUN mkdir spark && \
    curl ${SPARK} -o spark.tgz && \
    tar xzf spark.tgz --strip-components 1 -C spark && \
    rm spark.tgz

# Setup Conda environment
RUN python recommenders/scripts/generate_conda_file.py --name base --pyspark

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64" \
    PYSPARK_PYTHON="${HOME}/conda/bin/python" \
    PYSPARK_DRIVER_PYTHON="${HOME}/conda/bin/python" \
    SPARK_HOME="${HOME}/spark"


# GPU Stage
FROM nvidia/cuda:9.0-base AS gpu

ARG HOME
WORKDIR ${HOME}

COPY --from=base ${HOME} .

# Setup Conda environment
ENV PATH="${HOME}/conda/bin:${PATH}"
RUN python recommenders/scripts/generate_conda_file.py --name base --gpu


# Full Stage
FROM gpu AS full

ARG HOME
WORKDIR ${HOME}

COPY --from=pyspark ${HOME}/spark spark

# Setup Conda environment
RUN python recommenders/scripts/generate_conda_file.py --name base --gpu --pyspark

ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64" \
    PYSPARK_PYTHON="${HOME}/conda/bin/python" \
    PYSPARK_DRIVER_PYTHON="${HOME}/conda/bin/python" \
    SPARK_HOME="${HOME}/spark"


# Final Stage
FROM $ENV AS final

# Install Conda dependencies
RUN conda env update -f base.yaml && \
    conda clean -fay && \
    python -m ipykernel install --user --name 'python3' --display-name 'python3'

ARG HOME
WORKDIR ${HOME}/recommenders

EXPOSE 8888
CMD ["jupyter", "notebook"]

