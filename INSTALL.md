# Installation guide

## Requirements

- [Anaconda Python 3.6](https://conda.io/miniconda.html)
- Machine with GPU (optional)
- Machine with Spark (optional)

## Conda environments

As a pre-requisite, we may want to make sure that Conda is up-to-date:

    conda update conda

We have three different environments: Python CPU, Python GPU and PySpark. We provided a script to [generate a conda file](scripts/generate_conda_file.sh), depending of the environment we want to use.

To install each environment, first we need to generate a conda yml file and then install the environment. We can choose the environment name with the input `-n`. In the following examples, we provide a name example.

### Python CPU environment

To install the Python CPU environment:

    ./scripts/generate_conda_file.sh
    conda env create -n reco_bare -f conda_bare.yaml 

### Python GPU environment

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    ./scripts/generate_conda_file.sh --gpu
    conda env create -n reco_gpu -f conda_gpu.yaml 

### PySpark environment

To install the PySpark environment, which by default installs the CPU environment:

    ./scripts/generate_conda_file.sh --pyspark
    conda env create -n reco_pyspark -f conda_gpu.yaml

For this environment, we need to set the environment variables `PYSPARK_PYTHON` and `PYSPARK_DRIVER_PYTHON` to point to the conda python executable. In this [guide](https://conda.io/docs/user-guide/tasks/manage-environments.html#macos-and-linux), it is shown how these variables can be added every time the environment is activated.

### All environments

To install all three environments:

    ./scripts/generate_conda_file.sh  --gpu --pyspark
    conda env create -n reco_full -f conda_full.yaml

