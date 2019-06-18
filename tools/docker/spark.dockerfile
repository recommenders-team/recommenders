# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

FROM ubuntu:18.04

MAINTAINER "Microsoft Recommender Project <RecoDevTeam@service.microsoft.com>"

WORKDIR /root

ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64
ENV PATH /opt/conda/bin:$PATH
ENV PYSPARK_PYTHON /opt/conda/bin/python
ENV PYSPARK_DRIVER_PYTHON /opt/conda/bin/python
ENV SPARK_HOME /opt/spark

RUN apt-get update && \
    apt-get install -y curl git openjdk-8-jre

ARG SPARK="http://archive.apache.org/dist/spark/spark-2.3.1/spark-2.3.1-bin-hadoop2.7.tgz"
RUN mkdir /opt/spark && \
    curl ${SPARK} -o spark.tgz && \
    tar xzf spark.tgz --strip-components 1 -C /opt/spark

ARG ANACONDA="https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh"
RUN curl ${ANACONDA} -o anaconda.sh && \
    /bin/bash anaconda.sh -b -p /opt/conda && \
    rm anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ARG BRANCH="master"
RUN git clone https://github.com/microsoft/recommenders && \
    cd recommenders && \
    git checkout ${BRANCH} && \
    python scripts/generate_conda_file.py --name base --pyspark && \
    conda env update -f base.yaml && \
    python -m ipykernel install --user --name 'python3' --display-name 'python3'

EXPOSE 8888
ENV NOTEBOOK_CONFIG="/root/.jupyter/jupyter_notebook_config.py"
RUN mkdir /root/.jupyter && \
    echo "c.NotebookApp.token = ''" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.allow_root = True" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >> ${NOTEBOOK_CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >> ${NOTEBOOK_CONFIG}

WORKDIR /root/recommenders
CMD ["jupyter", "notebook"]