One possible way to use the repository is to run all the recommender utilities directly from a local copy of the source code (without building the package). This requires installing all the necessary dependencies from Anaconda and PyPI.

To this end we provide a script, [generate_conda_file.py](tools/generate_conda_file.py), to generate a conda-environment yaml file which you can use to create the target environment using Python 3.6 or 3.7 with all the correct dependencies.

Assuming the repo is cloned as `Recommenders` in the local system, to install **a default (Python CPU) environment**:

    cd Recommenders
    python tools/generate_conda_file.py
    conda env create -f reco_base.yaml

You can specify the environment name as well with the flag `-n`.

Click on the following menus to see how to install Python GPU and PySpark environments:

<details>
<summary><strong><em>Python GPU environment</em></strong></summary>

Assuming that you have a GPU machine, to install the Python GPU environment:

    cd Recommenders
    python tools/generate_conda_file.py --gpu
    conda env create -f reco_gpu.yaml

</details>

<details>
<summary><strong><em>PySpark environment</em></strong></summary>

To install the PySpark environment:

    cd Recommenders
    python tools/generate_conda_file.py --pyspark
    conda env create -f reco_pyspark.yaml

Additionally, if you want to test a particular version of spark, you may pass the `--pyspark-version` argument:

    python tools/generate_conda_file.py --pyspark-version 2.4.5

</details>

<details>
<summary><strong><em>Full (PySpark & Python GPU) environment</em></strong></summary>

With this environment, you can run both PySpark and Python GPU notebooks in this repository.
To install the environment:

    cd Recommenders
    python tools/generate_conda_file.py --gpu --pyspark
    conda env create -f reco_full.yaml

</details>
