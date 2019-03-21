# Setup guide

This document describes how to setup all the dependencies to run the notebooks in this repository in two different platforms:

* Linux Machine: Local or [Azure Data Science Virtual Machine (DSVM)](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/)
* [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/)

## Table of Contents

* [Compute environments](#compute-environments)
* [Setup guide for Local or DSVM](#setup-guide-for-local-or-dsvm)
  * [Setup Requirements](#setup-requirements)
  * [Dependencies setup](#dependencies-setup)
  * [Register the conda environment as a kernel in Jupyter](#Register-the-conda-environment-as-a-kernel-in-Jupyter)
  * [Troubleshooting for the DSVM](#troubleshooting-for-the-dsvm)
* [Setup guide for Azure Databricks](#setup-guide-for-azure-databricks)
  * [Requirements of Azure Databricks](#requirements-of-azure-databricks)
  * [Repository installation](#repository-installation)
  * [Troubleshooting Installation on Azure Databricks](#Troubleshooting-Installation-on-Azure-Databricks)
* [Prepare Azure Databricks for Operationalization](#prepare-azure-databricks-for-operationalization)

## Compute environments

Depending on the type of recommender system and the notebook that needs to be run, there are different computational requirements. Currently, this repository supports the following environments:

* Python CPU
* Python GPU
* PySpark

## Setup guide for Local or DSVM

### Requirements

* Machine running Linux, Windows Subsystem for Linux ([WSL](https://docs.microsoft.com/en-us/windows/wsl/about)) or macOS
* Anaconda with Python version >= 3.6.
  * This is pre-installed on Azure DSVM, for local setup [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is a quick way to get started.
* [Apache Spark](https://spark.apache.org/downloads.html) (this is only needed for the PySpark environment).

### Dependencies setup

We install the dependencies with Conda. As a pre-requisite, we want to make sure that Anaconda and the package manager Conda are both up to date:

```{shell}
conda update conda -n root
conda update anaconda
```

We provide a script, [generate_conda_file.py](scripts/generate_conda_file.py), to generate a conda file, depending of the environment we want to use. This will create the environment using the Python version 3.6 with all the correct dependencies.

To install each environment, first we need to generate a conda yaml file and then install the environment. We can specify the environment name with the input `-n`.

Click on the following menus to see more details:

<details>
<summary><strong><em>Python CPU environment</em></strong></summary>

Assuming the repo is cloned as `Recommenders` in the local system, to install the Python CPU environment:

    cd Recommenders
    python scripts/generate_conda_file.py
    conda env create -f reco_base.yaml 

</details>

<details>
<summary><strong><em>Python GPU environment</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    cd Recommenders
    python scripts/generate_conda_file.py --gpu
    conda env create -f reco_gpu.yaml 

</details>

<details>
<summary><strong><em>PySpark environment</em></strong></summary>

To install the PySpark environment, which by default installs the CPU environment:

    cd Recommenders
    python scripts/generate_conda_file.py --pyspark
    conda env create -f reco_pyspark.yaml

Additionally, if you want to test a particular version of spark, you may pass the --pyspark-version argument:

    python scripts/generate_conda_file.py --pyspark-version 2.4.0

**NOTE** - for a PySpark environment, we need to set the environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to point to the conda python executable.

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
    python scripts/generate_conda_file.py --gpu --pyspark
    conda env create -f reco_full.yaml

</details>

### Register the conda environment as a kernel in Jupyter

We can register our created conda environment to appear as a kernel in the Jupyter notebooks.

    conda activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

### Troubleshooting for the DSVM

* We found that there can be problems if the Spark version of the machine is not the same as the one in the conda file. You can use the option `--pyspark-version` to address this issue.
* When running Spark on a single local node it is possible to run out of disk space as temporary files are written to the user's home directory. To avoid this on a DSVM, we attached an additional disk to the DSVM and made modifications to the Spark configuration. This is done by including the following lines in the file at `/dsvm/tools/spark/current/conf/spark-env.sh`.

```{shell}
SPARK_LOCAL_DIRS="/mnt"
SPARK_WORKER_DIR="/mnt"
SPARK_WORKER_OPTS="-Dspark.worker.cleanup.enabled=true, -Dspark.worker.cleanup.appDataTtl=3600, -Dspark.worker.cleanup.interval=300, -Dspark.storage.cleanupFilesAfterExecutorExit=true"
```

## Setup guide for Azure Databricks

### Requirements

* Databricks Runtime version 4.3 (Apache Spark 2.3.1, Scala 2.11) or greater
* Python 3

An example of how to create an Azure Databricks workspace and an Apache Spark cluster within the workspace can be found from [here](https://docs.microsoft.com/en-us/azure/azure-databricks/quickstart-create-databricks-workspace-portal). To utilize deep learning models and GPUs, you may setup GPU-enabled cluster. For more details about this topic, please see [Azure Databricks deep learning guide](https://docs.azuredatabricks.net/applications/deep-learning/index.html).   

### Repository installation
You can setup the repository as a library on Databricks either manually or by running an [installation script](scripts/databricks_install.py). Both options assume you have access to a provisioned Databricks workspace and cluster and that you have appropriate permissions to install libraries.

<details>
<summary><strong><em>Quick install</em></strong></summary>

This option utilizes an installation script to do the setup, and it requires additional dependencies in the environment used to execute the script.

> To run the script, following **prerequisites** are required:
> * Setup CLI authentication for [Azure Databricks CLI (command-line interface)](https://docs.azuredatabricks.net/user-guide/dev-tools/databricks-cli.html#install-the-cli). Please find details about how to create a token and set authentication [here](https://docs.azuredatabricks.net/user-guide/dev-tools/databricks-cli.html#set-up-authentication). Very briefly, you can install and configure your environment with the following commands.
>
>     ```{shell}
>     conda activate reco-pyspark
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


Once you have confirmed the databricks cluster is *RUNNING*, install the modules within this repository with the following commands. 

```{shell}
cd Recommenders
python scripts/databricks_install.py <CLUSTER_ID>
```

The installation script has a number of options that can also deal with different databricks-cli profiles, install a version of the mmlspark library, or prepare the cluster for operationalization. For all options, please see:

```{shell}
python scripts/databricks_install.py -h
```

**Note** If you are planning on running through the sample code for operationalization [here](notebooks/05_operationalize/als_movie_o16n.ipynb), you need to prepare the cluster for operationalization. You can do so by adding an additional option to the script run. <CLUSTER_ID> is the same as that mentioned above, and can be identified by running `databricks clusters list` and selecting the appropriate cluster.

```{shell}
./scripts/databricks_install.py --prepare-o16n <CLUSTER_ID>
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
import reco_utils
```

### Troubleshooting Installation on Azure Databricks

* For the [reco_utils](reco_utils) import to work on Databricks, it is important to zip the content correctly. The zip has to be performed inside the Recommenders folder, if you zip directly above the Recommenders folder, it won't work.

## Prepare Azure Databricks for Operationalization

This repository includes an end-to-end example notebook that uses Azure Databricks to estimate a recommendation model using matrix factorization with Alternating Least Squares, writes pre-computed recommendations to Azure Cosmos DB, and then creates a real-time scoring service that retrieves the recommendations from Cosmos DB. In order to execute that [notebook](notebooks/05_operationalize/als_movie_o16n.ipynb), you must install the Recommenders repository as a library (as described above), **AND** you must also install some additional dependencies. With the *Quick install* method, you just need to pass an additional option to the [installation script](scripts/databricks_install.py).

<details>
<summary><strong><em>Quick install</em></strong></summary>

This option utilizes the installation script to do the setup. Just run the installation script
with an additional option. If you have already run the script once to upload and install the `Recommenders.egg` library, you can also add an `--overwrite` option:

```{shell}
scripts/databricks_install.py --overwrite --prepare-o16n <CLUSTER_ID>
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

1. Download the [appropriate jar](https://search.maven.org/remotecontent?filepath=com/microsoft/azure/azure-cosmosdb-spark_2.3.0_2.11/1.2.2/azure-cosmosdb-spark_2.3.0_2.11-1.2.2-uber.jar) from MAVEN. **NOTE** This is the appropriate jar for spark versions `2.3.X`, and is the appropriate version for the recommended Azure Databricks run-time detailed above.
2. Upload and install the jar by:
   1. Log into your `Azure Databricks` workspace
   2. Select the `Clusters` button on the left.
   3. Select the cluster on which you want to import the library.
   4. Select the `Upload` and `Jar` options, and click in the box that has the text `Drop JAR here` in it.
   5. Navigate to the downloaded `.jar` file, select it, and click `Open`.
   6. Click on `Install`.
   7. Restart the cluster.

</details>
