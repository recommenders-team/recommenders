# Setup guide 

In this guide we show how to setup all the dependencies to run the notebooks of this repo on an [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) and on [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/). 

<details>
<summary><strong><em>Click here to see the Table of Contents</em></strong></summary>
 
* [Compute environments](#compute-environments)
* [Setup guide for the DSVM](#setup-guide-for-the-dsvm)
  * [Requirements of the DSVM](#requirements-of-the-dsvm)
  * [Dependencies setup for the DSVM](#dependencies-setup-for-the-dsvm)
  * [Register the conda environment in Jupyter notebook](register-the-conda-environment-in-jupyter-notebook)
  * [Tests](#tests)
  * [Troubleshooting for the DSVM](#troubleshooting-for-the-dsvm)
* [Setup guide for Azure Databricks](#setup-guide-for-azure-databricks)
  * [Requirements of Azure Databricks](#requirements-of-azure-databricks)
  * [Repository upload](#repository-upload)
  * [Dependencies setup for Azure Databricks](#dependencies-setup-for-azure-databricks)
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

## Setup guide for the DSVM

### Requirements of the DSVM

- [Anaconda Python 3.6](https://conda.io/miniconda.html)
- The Python library dependencies can be found in this [script](scripts/generate_conda_file.sh).
- Machine with Spark (optional for Python environment but mandatory for PySpark environment).
- Machine with GPU (optional but desirable for computing acceleration).

### Dependencies setup for the DSVM

We install the dependencies with Conda. As a pre-requisite, we may want to make sure that Conda is up-to-date:

    conda update conda

We provided a script to [generate a conda file](scripts/generate_conda_file.sh), depending of the environment we want to use.

To install each environment, first we need to generate a conda yml file and then install the environment. We can specify the environment name with the input `-n`. Click on the following menus to see more details:

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

**NOTE** for this environment, we need to set the environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to point to the conda python executable.

For setting these variables every time the environment is activated, we can follow the steps of this [guide](https://conda.io/docs/user-guide/tasks/manage-environments.html#macos-and-linux). Assuming that we have installed the environment in `/anaconda/envs/reco_pyspark`, we create the file `/anaconda/envs/reco_pyspark/activate.d/env_vars.sh` and add:

```bash
#!/bin/sh
export PYSPARK_PYTHON=/anaconda/envs/reco_pyspark/bin/python
export PYSPARK_DRIVER_PYTHON=/anaconda/envs/reco_pyspark/bin/python
```

This will export the variables every time we do `source activate reco_pyspark`. To unset these variables when we deactivate the environment, we create the file `/anaconda/envs/reco_pyspark/deactivate.d/env_vars.sh` and add:

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

    source activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

### Tests

This project use unit, smoke and integration tests with Python files and notebooks. For more information, see a [quick introduction to unit, smoke and integration tests](https://miguelgfierro.com/blog/2018/a-beginners-guide-to-python-testing/). Click on the following menus to see more details:

<details>
<summary><strong><em>Unit tests</em></strong></summary>

Unit tests ensure that each class or function behaves as it should. Every time a developer makes a pull request to staging or master branch, a battery of unit tests is executed. To manually execute the unit tests in the different environments, first **make sure you are in the correct environment**.

For executing the Python unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not spark and not gpu"

For executing the Python unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not spark and not gpu"

For executing the Python GPU unit tests for the utilities:

    pytest tests/unit -m "not notebooks and not spark and gpu"

For executing the Python GPU unit tests for the notebooks:

    pytest tests/unit -m "notebooks and not spark and gpu"

For executing the PySpark unit tests for the utilities:

    pytest tests/unit -m "not notebooks and spark and not gpu"

For executing the PySpark unit tests for the notebooks:

    pytest tests/unit -m "notebooks and spark and not gpu"

</details>


<details>
<summary><strong><em>Smoke tests</em></strong></summary>

Smoke tests make sure that the system works and are executed just before the integration tests every night.

For executing the Python smoke tests:

    pytest tests/smoke -m "smoke and not spark and not gpu"

For executing the Python GPU smoke tests:

    pytest tests/smoke -m "smoke and not spark and gpu"

For executing the PySpark smoke tests:

    pytest tests/smoke -m "smoke and spark and not gpu"

</details>

<details>
<summary><strong><em>Integration tests</em></strong></summary>

Integration tests make sure that the program results are acceptable

For executing the Python integration tests:

    pytest tests/integration -m "integration and not spark and not gpu"

For executing the Python GPU integration tests:

    pytest tests/integration -m "integration and not spark and gpu"

For executing the PySpark integration tests:

    pytest tests/integration -m "integration and spark and not gpu"

</details>


### Troubleshooting for the DSVM

* We found that there could be problems if the Spark version of the machine is not the same as the one in the conda file. You will have to adapt the conda file to your machine. 

## Setup guide for Azure Databricks

### Requirements of Azure Databricks
* Runtime version 4.3 (Apache Spark 2.3.1, Scala 2.11)
* Python 3

### Repository upload
We need to zip and upload the repository to be used in Databricks, the steps are the following:
* Clone Microsoft Recommenders repo in your local computer.
* Zip the content inside the root folder:
```
cd Recommenders
zip -r Recommenders.zip .
```
* Once your cluster has started, go to the Databricks home workspace, then go to your user and press import.
* In the next menu there is an option to import a library, it says: `To import a library, such as a jar or egg, click here`. Press click here.
* Then, at the first drop-down menu, mark the option `Upload Python egg or PyPI`.
* Then press on `Drop library egg here to upload` and select the the file `Recommenders.zip` you just created.
* Then press `Create library`. This will upload the zip and make it available in your workspace.
* Finally, in the next menu, attach the library to your cluster.

To make sure it works, you can now create a new notebook and import the utilities:
```
import reco_utils
```

### Dependencies setup for Azure Databricks
The dependencies has to be manually installed in the cluster, they can be found on [this script](scripts/generate_conda_file.sh).

### Troubleshooting for Azure Databricks
* For the [utilities](reco_utils) to work on Databricks, it is important to zip the content correctly. The zip has to be performed inside the root folder, if you zip directly the root folder, it won't work.

