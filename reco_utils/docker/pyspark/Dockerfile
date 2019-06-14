# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
ARG BASE_CONTAINER=jupyter/pyspark-notebook
FROM $BASE_CONTAINER

LABEL maintainer="Microsoft Recommender Project <RecoDevTeam@service.microsoft.com>"

ARG WORK_PATH=/home/$NB_USER/work
ARG RECO_PATH=$WORK_PATH/recommenders
WORKDIR $WORK_PATH

ARG ENVIRONMENT="pyspark"

USER root

# Clone repo & discard notebooks that cannot be run in this environment
# RUN git clone --single-branch --branch only_pyspark https://github.com/Microsoft/Recommenders.git 
RUN git clone https://github.com/microsoft/recommenders.git 

# Install with the conda yaml file
RUN if ["$ENVIRONMENT" = "pyspark"]; then \
        python recommenders/scripts/generate_conda_file.py --pyspark --name=reco; \
    elif ["$ENVIRONMENT" = "gpu"]; then \
        python recommenders/scripts/generate_conda_file.py --gpu --name=reco; \
    else \
        python recommenders/scripts/generate_conda_file.py --pyspark --gpu --name=reco; \
    fi
RUN conda env create -f reco.yaml 

# # Install the needed packages
# RUN conda install --quiet --yes \
#     "mock==2.0.0" \
#     "fastparquet>=0.1.6" \
#     "gitpython>=2.1.8" \
#     "numpy>=1.13.3" \
#     "pymongo>=3.6.1" \
#     "pytest>=3.6.4" \
#     "scikit-surprise>=1.0.6" \
#     "lightgbm==2.2.1" 

# RUN pip install \
#     "azureml-sdk[notebooks,contrib]==1.0.18" \
#     "idna==2.7" \
#     "memory-profiler>=0.54.0" \
#     "papermill>=0.15.0" \
#     "pydocumentdb>=2.3.3" \
#     "tqdm==4.31.1"

RUN fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER && \
    fix-permissions $RECO_PATH
