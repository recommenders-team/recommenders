# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Base image is the one that is from Jupyter notebook Dockerhub
# Default environment value is "base", and accepted values are "pyspark" and "tensorflow", for
# Spark and deep learning environment, respectively.
ARG ENVIRONMENT='datascience'

FROM jupyter/$ENVIRONMENT-notebook
LABEL maintainer="Microsoft Recommender Project <RecoDevTeam@service.microsoft.com>"

USER root

# Specify working directory
ARG WORK_PATH=/home/$NB_USER/work
ARG RECO_PATH=$WORK_PATH/recommenders
WORKDIR $WORK_PATH

# Clone the repo
RUN git clone https://github.com/microsoft/recommenders.git 

# Config conda channel
RUN conda config --env --add channels conda-forge && \
    conda config --env --add channels pytorch && \
    conda config --env --add channels fastai 

# Install the base packages
RUN conda install --quiet --yes \
    "mock==2.0.0" \
    "fastparquet>=0.1.6" \
    "gitpython>=2.1.8" \
    "numpy>=1.13.3" \
    "pymongo>=3.6.1" \
    "pytest>=3.6.4" \
    "scikit-surprise>=1.0.6" \
    "lightgbm==2.2.1" 

RUN pip install \
    "azureml-sdk[notebooks,contrib]==1.0.18" \
    "idna==2.7" \
    "memory-profiler>=0.54.0" \
    "papermill>=0.15.0" \
    "pydocumentdb>=2.3.3" \
    "tqdm==4.31.1"

# Install conda & pip packages
RUN if ["$ENVIRONMENT" = "pyspark"]; then \
        # Install the needed packages for pyspark environment
        conda install --quiet --yes \
        "pyarrow>=0.8.0" \
        "pyspark==2.3.1"; \
    elif ["$ENVIRONMENT" = "tensorflow"]; then \
        # Install the needed packages for GPU environment
        conda install --quiet --yes \
        "numba>=0.38.1" \
        "pytorch>=1.0.0" \
        "tensorflow-gpu==1.12.0"; \
    fi

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    fix-permissions $RECO_PATH
