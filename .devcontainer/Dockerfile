ARG PYTHON_VERSION
FROM mcr.microsoft.com/vscode/devcontainers/python:${PYTHON_VERSION}

ENV DEBIAN_FRONTEND="noninteractive"
ENV JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
RUN apt-get update && \
    apt-get -y install --no-install-recommends software-properties-common && \
    apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main' && \
    apt-get update && \
    apt-get -y install --no-install-recommends openjdk-8-jre
