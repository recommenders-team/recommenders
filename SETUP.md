# Setup guide

In this guide we show how to setup all the dependencies to run the notebooks of this repo. 

Three environments are supported to run the notebooks in the repo:

* Python CPU
* Python GPU
* PySpark

## Requirements

- [Anaconda Python 3.6](https://conda.io/miniconda.html)
- The Python library dependencies can be found in this [script](scripts/generate_conda_file.sh).
- Machine with Spark (optional for Python environment but mandatory for PySpark environment).
- Machine with GPU (optional but desirable for computing acceleration).

## Conda environments

As a pre-requisite, we may want to make sure that Conda is up-to-date:

    conda update conda

We provided a script to [generate a conda file](scripts/generate_conda_file.sh), depending of the environment we want to use.

To install each environment, first we need to generate a conda yml file and then install the environment. We can specify the environment name with the input `-n`. In the following examples, we provide a name example.

### Python CPU environment

Assuming the repo is cloned as `Recommenders` in the local system, to install the Python CPU environment:

    cd Recommenders
    ./scripts/generate_conda_file.sh
    conda env create -n reco_bare -f conda_bare.yaml 

### Python GPU environment

Assuming that you have a GPU machine, to install the Python GPU environment, which by default installs the CPU environment:

    cd Recommenders
    ./scripts/generate_conda_file.sh --gpu
    conda env create -n reco_gpu -f conda_gpu.yaml 

### PySpark environment

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

### All environments

To install all three environments:

    cd Recommenders
    ./scripts/generate_conda_file.sh  --gpu --pyspark
    conda env create -n reco_full -f conda_full.yaml

### Register the conda environment in Jupyter notebook

We can register our created conda environment to appear as a kernel in the Jupyter notebooks. 

    source activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"

## Troubleshooting

* We found that there could be problems if the Spark version of the machine is not the same as the one in the conda file. You will have to adapt the conda file to your machine. 

