ARG PYTHON_VERSION
FROM mcr.microsoft.com/vscode/devcontainers/python:${PYTHON_VERSION}

ARG REMOTE_USER
USER ${REMOTE_USER}

ENV DEBIAN_FRONTEND="noninteractive" \
    JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64" \
    PYSPARK_PYTHON="/usr/local/bin/python" \
    PYSPARK_DRIVER_PYTHON="/usr/local/bin/python"

RUN apt-get update && \
    apt-get -y install --no-install-recommends software-properties-common && \
    apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main' && \
    apt-get update && \
    apt-get -y install --no-install-recommends \
      openjdk-8-jre \
      cmake

# Setup Jupyter Notebook
ENV NOTEBOOK_CONFIG="${HOME}/.jupyter/jupyter_notebook_config.py"
RUN mkdir ${HOME}/.jupyter && \
    echo "c.NotebookApp.open_browser=False" >> ${NOTEBOOK_CONFIG} && \
    echo "c.NotebookApp.allow_origin='*'" >> ${NOTEBOOK_CONFIG}

EXPOSE 8888