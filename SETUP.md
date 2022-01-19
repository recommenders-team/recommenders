# Setup guide

This document describes how to setup all the dependencies to run the notebooks in this repository in following platforms:

* Local (Linux, MacOS or Windows) or [DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (Linux or Windows)
* [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/)
* Docker container

## Table of Contents

  - [Compute environments](#compute-environments)
  - [Setup guide for Local or DSVM](#setup-guide-for-local-or-dsvm)
    - [Requirements](#requirements)
    - [Dependencies setup](#dependencies-setup)
    - [Using a virtual environment](#using-a-virtual-environment)
    - [Register the environment as a kernel in Jupyter](#register-the-environment-as-a-kernel-in-jupyter)
    - [Troubleshooting for the DSVM](#troubleshooting-for-the-dsvm)
  - [Setup guide for Azure Databricks](#setup-guide-for-azure-databricks)
    - [Requirements of Azure Databricks](#requirements-1)
    - [Installation from PyPI](#installation-from-pypi)
    - [Dependencies setup](#dependencies-setup-1)
    - [Confirm Installation](#confirm-installation)
    - [Troubleshooting Installation on Azure Databricks](#troubleshooting-installation-on-azure-databricks)
    - [Prepare Azure Databricks for Operationalization](#prepare-azure-databricks-for-operationalization)
  - [Setup guide for Docker](#setup-guide-for-docker)
  - [Setup guide for making a release](#setup-guide-for-making-a-release)

## Compute environments

Depending on the type of recommender system and the notebook that needs to be run, there are different computational requirements.
Currently, this repository supports **Python CPU**, **Python GPU** and **PySpark**.


## Setup guide for Local or DSVM

There are different ways one may use the recommenders utilities. The most convenient one is probably by installing the `recommenders` package from [PyPI](https://pypi.org).

Another way is to build a docker image and use the functions inside a [docker container](#setup-guide-for-docker).

Another alternative is to run all the recommender utilities directly from a local copy of the source code. This requires installing all the necessary dependencies from Anaconda and PyPI. For instructions on how to do this, see [this guide](conda.md).

### Requirements

* A machine running Linux, MacOS or Windows
* An optional requirement is Anaconda with Python version >= 3.6
  * This is pre-installed on Azure DSVM such that one can run the following steps directly. To setup on your local machine, [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.
  
  Alternatively a [virtual environment](#using-a-virtual-environment) can be used instead of Anaconda.
* [Apache Spark](https://spark.apache.org/downloads.html) (this is only needed for the PySpark environment).

### Dependencies setup

As a pre-requisite to installing the dependencies, if using Conda, make sure that Anaconda and the package manager Conda are both up to date:

```{shell}
conda update conda -n root
conda update anaconda        # use 'conda install anaconda' if the package is not installed
```

If using venv or virtualenv, see [these instructions](#using-a-virtual-environment).

**NOTE** the `xlearn` package has dependency on `cmake`. If one uses the `xlearn` related notebooks or scripts, make sure `cmake` is installed in the system. The easiest way to install on Linux is with apt-get: `sudo apt-get install -y build-essential cmake`. Detailed instructions for installing `cmake` from source can be found [here](https://cmake.org/install/).

**NOTE** the models from Cornac require installation of `libpython` i.e. using `sudo apt-get install -y libpython3.6` or `libpython3.7`, depending on the version of Python.

**NOTE** Spark requires Java version 8 or 11. We support Spark versions 3.0 and 3.1, but versions 2.4+ with Java version 8 may also work. 

<details> 
<summary><strong><em>Install Java on MacOS</em></strong></summary>
  
To install e.g. Java 8 on MacOS using [asdf](https://github.com/halcyon/asdf-java):

    brew install asdf
    asdf plugin add Java
    asdf install java adoptopenjdk-8.0.265+1
    asdf global java adoptopenjdk-8.0.265+1
    . ~/.asdf/plugins/java/set-java-home.zsh

</details>


Then, we need to set the environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to point to the conda python executable.

Click on the following menus to see details:
<details>
<summary><strong><em>Set PySpark environment variables on Linux or MacOS</em></strong></summary>

If you use conda, to set these variables every time the environment is activated, you can follow the steps of this [guide](https://conda.io/docs/user-guide/tasks/manage-environments.html#macos-and-linux).

First, assuming that the environment is called `reco_pyspark`, get the path where the environment is installed:

    RECO_ENV=$(conda env list | grep reco_pyspark | awk '{print $NF}')
    mkdir -p $RECO_ENV/etc/conda/activate.d
    mkdir -p $RECO_ENV/etc/conda/deactivate.d

Then, create the file `$RECO_ENV/etc/conda/activate.d/env_vars.sh` and add:

```bash
#!/bin/sh
RECO_ENV=$(conda env list | grep reco_pyspark | awk '{print $NF}')
export PYSPARK_PYTHON=$RECO_ENV/bin/python
export PYSPARK_DRIVER_PYTHON=$RECO_ENV/bin/python
unset SPARK_HOME
```

This will export the variables every time we do `conda activate reco_pyspark`. To unset these variables when we deactivate the environment, create the file `$RECO_ENV/etc/conda/deactivate.d/env_vars.sh` and add:

```bash
#!/bin/sh
unset PYSPARK_PYTHON
unset PYSPARK_DRIVER_PYTHON
```

</details>

<details><summary><strong><em>Set PySpark environment variables on Windows</em></strong></summary>

To set these variables every time the environment is activated, we can follow the steps of this [guide](https://conda.io/docs/user-guide/tasks/manage-environments.html#windows).
First, get the path of the environment `reco_pyspark` is installed:

    for /f "delims=" %A in ('conda env list ^| grep reco_pyspark ^| awk "{print $NF}"') do set "RECO_ENV=%A"

Then, create the file `%RECO_ENV%\etc\conda\activate.d\env_vars.bat` and add:

    @echo off
    for /f "delims=" %%A in ('conda env list ^| grep reco_pyspark ^| awk "{print $NF}"') do set "RECO_ENV=%%A"
    set PYSPARK_PYTHON=%RECO_ENV%\python.exe
    set PYSPARK_DRIVER_PYTHON=%RECO_ENV%\python.exe
    set SPARK_HOME_BACKUP=%SPARK_HOME%
    set SPARK_HOME=
    set PYTHONPATH_BACKUP=%PYTHONPATH%
    set PYTHONPATH=

This will export the variables every time we do `conda activate reco_pyspark`.
To unset these variables when we deactivate the environment,
create the file `%RECO_ENV%\etc\conda\deactivate.d\env_vars.bat` and add:

    @echo off
    set PYSPARK_PYTHON=
    set PYSPARK_DRIVER_PYTHON=
    set SPARK_HOME=%SPARK_HOME_BACKUP%
    set SPARK_HOME_BACKUP=
    set PYTHONPATH=%PYTHONPATH_BACKUP%
    set PYTHONPATH_BACKUP=

</details>


### Using a virtual environment

It is straightforward to install the recommenders package within a [virtual environment](https://docs.python.org/3/library/venv.html). However, setting up CUDA for use with a GPU can be cumbersome. We thus
recommend setting up [Nvidia docker](https://github.com/NVIDIA/nvidia-docker) and running the virtual environment within a container, as the most convenient way to do this.  
In the following `3.6` should be replaced with the Python version you are using and `8` should be replaced with the appropriate Java version. 

    # Start docker daemon if not running
    sudo dockerd &
    # Pull the image from the Nvidia docker hub (https://hub.docker.com/r/nvidia/cuda) that is suitable for your system
    # E.g. for Ubuntu 18.04 do
    sudo docker run --gpus all -it --rm nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu18.04

    # Within the container: 

    apt-get -y update
    apt-get -y install python3.6
    apt-get -y install python3-pip
    apt-get -y install python3.6-venv
    apt-get -y install libpython3.6-dev
    apt-get -y install cmake
    apt-get install -y libgomp1 openjdk-8-jre
    export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
    
    python3.6 -m venv --system-site-packages /venv
    source /venv/bin/activate
    pip install --upgrade pip
    pip install --upgrade setuptools

    export SPARK_HOME=/venv/lib/python3.6/site-packages/pyspark
    export PYSPARK_DRIVER_PYTHON=/venv/bin/python
    export PYSPARK_PYTHON=/venv/bin/python

    pip install recommenders[all]

If you prefer to use [virtualenv](https://virtualenv.pypa.io/en/latest/index.html#) instead of venv, you may follow the above steps, except you will need to replace the line

`apt-get -y install python3.6-venv` 

with 

`python3.6 -m pip install --user virtualenv`

and the line

`python3.6 -m venv --system-site-packages /venv`

with

`python3.6 -m virtualenv /venv`


### Register the environment as a kernel in Jupyter

We can register our conda or virtual environment to appear as a kernel in the Jupyter notebooks. After activating the environment (`my_env_name`) do

    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

If you are using the DSVM, you can [connect to JupyterHub](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro#jupyterhub-and-jupyterlab) by browsing to `https://your-vm-ip:8000`.


### Troubleshooting for the DSVM

* We found that there can be problems if the Spark version of the machine is not the same as the one in the [conda file](conda.md). You can use the option `--pyspark-version` to address this issue.

* When running Spark on a single local node it is possible to run out of disk space as temporary files are written to the user's home directory. To avoid this on a DSVM, we attached an additional disk to the DSVM and made modifications to the Spark configuration. This is done by including the following lines in the file at `/dsvm/tools/spark/current/conf/spark-env.sh`.

```{shell}
SPARK_LOCAL_DIRS="/mnt"
SPARK_WORKER_DIR="/mnt"
SPARK_WORKER_OPTS="-Dspark.worker.cleanup.enabled=true, -Dspark.worker.cleanup.appDataTtl=3600, -Dspark.worker.cleanup.interval=300, -Dspark.storage.cleanupFilesAfterExecutorExit=true"
```

* Another source of problems is when the variable `SPARK_HOME` is not set correctly. In the Azure DSVM, `SPARK_HOME` is by default `/dsvm/tools/spark/current`. We need to unset it: 
```
unset SPARK_HOME
```

* We found that there might be conflicts between the current MMLSpark jars available in the DSVM and the ones used by the library. In that case, it is better to remove those jars and rely on loading them from Maven or other repositories made available by MMLSpark team.

```
cd /dsvm/tools/spark/current/jars
sudo rm -rf Azure_mmlspark-0.12.jar com.microsoft.cntk_cntk-2.4.jar com.microsoft.ml.lightgbm_lightgbmlib-2.0.120.jar
```

## Setup guide for Azure Databricks

### Requirements

* Databricks Runtime version >= 7, <= 9 (Apache Spark >= 3.0, <= 3.1, Scala 2.12)
* Python 3.6 or 3.7

Earlier versions of Databricks or Spark may work but this is not guaranteed.
An example of how to create an Azure Databricks workspace and an Apache Spark cluster within the workspace can be found from [here](https://docs.microsoft.com/en-us/azure/azure-databricks/quickstart-create-databricks-workspace-portal). To utilize deep learning models and GPUs, you may setup GPU-enabled cluster. For more details about this topic, please see [Azure Databricks deep learning guide](https://docs.azuredatabricks.net/applications/deep-learning/index.html).

### Installation from PyPI

The `recommenders` package can be installed with core dependencies for utilities and CPU-based algorithms.
This is done from the _Libraries_ link at the cluster, selecting the option to import a library and selecting _PyPI_ in the menu.  
For installations with more dependencies, see the steps below.

### Dependencies setup

You can setup the repository as a library on Databricks either manually or by running an [installation script](tools/databricks_install.py). Both options assume you have access to a provisioned Databricks workspace and cluster and that you have appropriate permissions to install libraries.

<details>
<summary><strong><em>Quick install</em></strong></summary>

This option utilizes an installation script to do the setup, and it requires additional dependencies in the environment used to execute the script.

> To run the script, following **prerequisites** are required:
> * Setup CLI authentication for [Azure Databricks CLI (command-line interface)](https://docs.azuredatabricks.net/user-guide/dev-tools/databricks-cli.html#install-the-cli). Please find details about how to create a token and set authentication [here](https://docs.azuredatabricks.net/user-guide/dev-tools/databricks-cli.html#set-up-authentication). Very briefly, you can install and configure your environment with the following commands.
>
>     ```{shell}
>     conda activate reco_pyspark
>     databricks configure --token
>     ```
>
> * Get the target **cluster id** and **start** the cluster if its status is *TERMINATED*.
>   * You can get the cluster id from the databricks CLI with:
>        ```{shell}
>        databricks clusters list
>        ```
>   * If required, you can start the cluster with:
>        ```{shell}
>        databricks clusters start --cluster-id <CLUSTER_ID>`
>        ```

The installation script has a number of options that can also deal with different databricks-cli profiles, install a version of the mmlspark library, overwrite the libraries, or prepare the cluster for operationalization. For all options, please see:

```{shell}
python tools/databricks_install.py -h
```

Once you have confirmed the databricks cluster is *RUNNING*, install the modules within this repository with the following commands.

```{shell}
cd Recommenders
python tools/databricks_install.py <CLUSTER_ID>
```

**Note** If you are planning on running through the sample code for operationalization [here](examples/05_operationalize/als_movie_o16n.ipynb), you need to prepare the cluster for operationalization. You can do so by adding an additional option to the script run. <CLUSTER_ID> is the same as that mentioned above, and can be identified by running `databricks clusters list` and selecting the appropriate cluster.

```{shell}
python tools/databricks_install.py --prepare-o16n <CLUSTER_ID>
```

See below for details.

</details>

<details>
<summary><strong><em>Manual setup</em></strong></summary>

To install the repo manually onto Databricks, follow the steps:

1. Clone the Microsoft Recommenders repository to your local computer.
2. Zip the contents inside the Recommenders folder (Azure Databricks requires compressed folders to have the `.egg` suffix, so we don't use the standard `.zip`):

    ```{shell}
    cd Recommenders
    zip -r Recommenders.egg .
    ```

3. Once your cluster has started, go to the Databricks workspace, and select the `Home` button.
4. Your `Home` directory should appear in a panel. Right click within your directory, and select `Import`.
5. In the pop-up window, there is an option to import a library, where it says: `(To import a library, such as a jar or egg, click here)`. Select `click here`.
6. In the next screen, select the option `Upload Python Egg or PyPI` in the first menu.
7. Next, click on the box that contains the text `Drop library egg here to upload` and use the file selector to choose the `Recommenders.egg` file you just created, and select `Open`.
8. Click on the `Create library`. This will upload the egg and make it available in your workspace.
9. Finally, in the next menu, attach the library to your cluster.

</details>

### Confirm Installation

After installation, you can now create a new notebook and import the utilities from Databricks in order to confirm that the import worked.

```{python}
import recommenders
```

### Troubleshooting Installation on Azure Databricks

* For the [recommenders](recommenders) import to work on Databricks, it is important to zip the content correctly. The zip has to be performed inside the Recommenders folder, if you zip directly above the Recommenders folder, it won't work.

### Prepare Azure Databricks for Operationalization

This repository includes an end-to-end example notebook that uses Azure Databricks to estimate a recommendation model using matrix factorization with Alternating Least Squares, writes pre-computed recommendations to Azure Cosmos DB, and then creates a real-time scoring service that retrieves the recommendations from Cosmos DB. In order to execute that [notebook](examples/05_operationalize/als_movie_o16n.ipynb), you must install the Recommenders repository as a library (as described above), **AND** you must also install some additional dependencies. With the *Quick install* method, you just need to pass an additional option to the [installation script](tools/databricks_install.py).

<details>
<summary><strong><em>Quick install</em></strong></summary>

This option utilizes the installation script to do the setup. Just run the installation script
with an additional option. If you have already run the script once to upload and install the `Recommenders.egg` library, you can also add an `--overwrite` option:

```{shell}
python tools/databricks_install.py --overwrite --prepare-o16n <CLUSTER_ID>
```

This script does all of the steps described in the *Manual setup* section below.

</details>

<details>
<summary><strong><em>Manual setup</em></strong></summary>

You must install three packages as libraries from PyPI:

* `azure-cli==2.0.56`
* `azureml-sdk[databricks]==1.0.8`
* `pydocumentdb==2.3.3`

You can follow instructions [here](https://docs.azuredatabricks.net/user-guide/libraries.html#install-a-library-on-a-cluster) for details on how to install packages from PyPI.

Additionally, you must install the [spark-cosmosdb connector](https://docs.databricks.com/spark/latest/data-sources/azure/cosmosdb-connector.html) on the cluster. The easiest way to manually do that is to:


1. Download the [appropriate jar](https://search.maven.org/remotecontent?filepath=com/azure/cosmos/spark/azure-cosmos-spark_3-1_2-12/4.3.1/azure-cosmos-spark_3-1_2-12-4.3.1.jar) from MAVEN. **NOTE** This is the appropriate jar for spark versions `3.1.X`, and is the appropriate version for the recommended Azure Databricks run-time detailed above. See the [Databricks installation script](https://github.com/microsoft/recommenders/blob/main/tools/databricks_install.py#L45) for other Databricks runtimes.
2. Upload and install the jar by:
   1. Log into your `Azure Databricks` workspace
   2. Select the `Clusters` button on the left.
   3. Select the cluster on which you want to import the library.
   4. Select the `Upload` and `Jar` options, and click in the box that has the text `Drop JAR here` in it.
   5. Navigate to the downloaded `.jar` file, select it, and click `Open`.
   6. Click on `Install`.
   7. Restart the cluster.

</details>


## Setup guide for Docker

A [Dockerfile](tools/docker/Dockerfile) is provided to build images of the repository to simplify setup for different environments. You will need [Docker Engine](https://docs.docker.com/install/) installed on your system.

*Note: `docker` is already available on Azure Data Science Virtual Machine*

See guidelines in the Docker [README](tools/docker/README.md) for detailed instructions of how to build and run images for different environments.

Example command to build and run Docker image with base CPU environment.
```{shell}
DOCKER_BUILDKIT=1 docker build -t recommenders:cpu --build-arg ENV="cpu" --build-arg VIRTUAL_ENV="conda" .
docker run -p 8888:8888 -d recommenders:cpu
```

You can then open the Jupyter notebook server at http://localhost:8888

## Setup guide for making a release

The process of making a new release and publishing it to pypi is as follows:

First make sure that the tag that you want to add, e.g. `0.6.0`, is added in [`recommenders.py/__init__.py`](recommenders.py/__init__.py). Follow the [contribution guideline](CONTRIBUTING.md) to add the change.

1. Make sure that the code in main passes all the tests (unit and nightly tests).
1. Create a tag with the version number: e.g. `git tag -a 0.6.0 -m "Recommenders 0.6.0"`.
1. Push the tag to the remote server: `git push origin 0.6.0`.
1. When the new tag is pushed, a release pipeline is executed. This pipeline runs all the tests again (unit, smoke and integration), 
generates a wheel and a tar.gz which are uploaded to a [GitHub draft release](https://github.com/microsoft/recommenders/releases).
1. Fill up the draft release with all the recent changes in the code.
1. Download the wheel and tar.gz locally, these files shouldn't have any bug, since they passed all the tests.
1. Install twine: `pip install twine`
1. Publish the wheel and tar.gz to pypi: `twine upload recommenders*`
