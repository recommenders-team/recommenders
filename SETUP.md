# Setup guide 

In this guide we show how to setup all the dependencies to run the notebooks of this repo on a local environment or [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) and on [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/). 

## Table of Contents
 
* [Compute environments](#compute-environments)
* [Setup guide for Local or DSVM](#setup-guide-for-local-or-dsvm)
  * [Setup Requirements](#setup-requirements)
  * [Dependencies setup](#dependencies-setup)
  * [Register the conda environment in Jupyter notebook](#register-the-conda-environment-in-jupyter-notebook)
  * [Troubleshooting for the DSVM](#troubleshooting-for-the-dsvm)
* [Setup guide for Azure Databricks](#setup-guide-for-azure-databricks)
  * [Requirements of Azure Databricks](#requirements-of-azure-databricks)
  * [Repository upload](#repository-upload)
  * [Troubleshooting for Azure Databricks](#troubleshooting-for-azure-databricks)
</details>

## Compute environments

We have different compute environments, depending on the kind of machine

Environments supported to run the notebooks on the DSVM:
* Python CPU
* Python GPU
* PySpark

Environments supported to run the notebooks on Azure Databricks:
* PySpark

## Setup guide for Local or DSVM

### Setup Requirements

- Anaconda with Python version >= 3.6. [Miniconda](https://conda.io/miniconda.html) is the fastest way to get started.
- The Python library dependencies can be found in this [script](scripts/generate_conda_file.sh).
- Machine with Spark (optional for Python environment but mandatory for PySpark environment).

### Dependencies setup

We install the dependencies with Conda. As a pre-requisite, we may want to make sure that Conda is up-to-date:

    conda update anaconda

We provided a script to [generate a conda file](scripts/generate_conda_file.sh), depending of the environment we want to use. This will create the environment using the Python version 3.6 with all the correct dependencies.

To install each environment, first we need to generate a conda yaml file and then install the environment. We can specify the environment name with the input `-n`. 

Click on the following menus to see more details:

<details>
<summary><strong><em>Python CPU environment</em></strong></summary>

Assuming the repo is cloned as `Recommenders` in the local system, to install the Python CPU environment:

    cd Recommenders
    ./scripts/generate_conda_file.sh
    conda env create -n reco_bare -f conda_bare.yaml 

</details>


<details>
<summary><strong><em>Python GPU environment</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    cd Recommenders
    ./scripts/generate_conda_file.sh --gpu
    conda env create -n reco_gpu -f conda_gpu.yaml 

</details>

<details>
<summary><strong><em>PySpark environment</em></strong></summary>

To install the PySpark environment, which by default installs the CPU environment:

    cd Recommenders
    ./scripts/generate_conda_file.sh --pyspark
    conda env create -n reco_pyspark -f conda_pyspark.yaml

**NOTE** - for this environment, we need to set the environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to point to the conda python executable.

To set these variables every time the environment is activated, we can follow the steps of this [guide](https://conda.io/docs/user-guide/tasks/manage-environments.html#macos-and-linux). Assuming that we have installed the environment in `/anaconda/envs/reco_pyspark`, we create the file `/anaconda/envs/reco_pyspark/etc/conda/activate.d/env_vars.sh` and add:

```bash
#!/bin/sh
export PYSPARK_PYTHON=/anaconda/envs/reco_pyspark/bin/python
export PYSPARK_DRIVER_PYTHON=/anaconda/envs/reco_pyspark/bin/python
```

This will export the variables every time we do `conda activate reco_pyspark`. To unset these variables when we deactivate the environment, we create the file `/anaconda/envs/reco_pyspark/etc/conda/deactivate.d/env_vars.sh` and add:

```bash
#!/bin/sh
unset PYSPARK_PYTHON
unset PYSPARK_DRIVER_PYTHON
```
</details>

<details>
<summary><strong><em>All environments</em></strong></summary>

To install all three environments:

    cd Recommenders
    ./scripts/generate_conda_file.sh  --gpu --pyspark
    conda env create -n reco_full -f conda_full.yaml

</details>


### Register the conda environment in Jupyter notebook

We can register our created conda environment to appear as a kernel in the Jupyter notebooks. 

    conda activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"


### Troubleshooting for the DSVM

* We found that there could be problems if the Spark version of the machine is not the same as the one in the conda file. You will have to adapt the conda file to your machine. 
* When running Spark on a single local node it is possible to run out of disk space as temporary files are written to the user's home directory. To avoid this we attached an additional disk to the DSVM and made modifications to the Spark configuration. This is done by including the following lines in the file at `/dsvm/tools/spark/current/conf/spark-env.sh`.
```
SPARK_LOCAL_DIRS="/mnt"
SPARK_WORKER_DIR="/mnt"
SPARK_WORKER_OPTS="-Dspark.worker.cleanup.enabled=true, -Dspark.worker.cleanup.appDataTtl=3600, -Dspark.worker.cleanup.interval=300, -Dspark.storage.cleanupFilesAfterExecutorExit=true"
```

## Setup guide for Azure Databricks

### Requirements of Azure Databricks
* Runtime version 4.1 (Apache Spark 2.3.0, Scala 2.11)
* Python 3

### Repository upload
We need to zip and upload the repository to be used in Databricks, the steps are the following:
* Clone Microsoft Recommenders repo in your local computer.
* Zip the contents inside the Recommenders folder (Azure Databricks requires compressed folders to have the .egg suffix, so we don't use the standard .zip):
```
cd Recommenders
zip -r Recommenders.egg .
```
* Once your cluster has started, go to the Databricks home workspace, then go to your user and press import.
* In the next menu there is an option to import a library, it says: `To import a library, such as a jar or egg, click here`. Press click here.
* Then, at the first drop-down menu, mark the option `Upload Python egg or PyPI`.
* Then press on `Drop library egg here to upload` and select the the file `Recommenders.egg` you just created.
* Then press `Create library`. This will upload the zip and make it available in your workspace.
* Finally, in the next menu, attach the library to your cluster.

To make sure it works, you can now create a new notebook and import the utilities:
```
import reco_utils
```

### Troubleshooting for Azure Databricks
* For the [reco_utils](reco_utils) import to work on Databricks, it is important to zip the content correctly. The zip has to be performed inside the Recommenders folder, if you zip directly above the Recommenders folder, it won't work.

