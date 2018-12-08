# Recommenders 

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learnings on four key tasks: 
1. [Prepare Data](notebooks/01_prepare_data/README.md): Preparing and loading data for each recommender algorithm
2. [Model](notebooks/02_model/README.md): Building models using various recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)), Singular Value Decomposition ([SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)), etc.
3. [Evaluate](notebooks/03_evaluate/README.md): Evaluating algorithms with offline metrics
4. [Operationalize](notebooks/04_operationalize/README.md): Operationalizing models in a production environment on Azure

Several utilities are provided in [reco_utils](reco_utils) to support common tasks such as loading datasets in the format expected by different algorithms, evaluating model outputs, and splitting train/test data. Implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications.

## Getting Started
Please see the [setup guide](SETUP.md) for more details on setting up your machine locally, on Spark, or on [Azure Databricks](/SETUP.md#setup-guide-for-azure-databricks). 

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
2. Clone the repository
    ```
    git clone https://github.com/Microsoft/Recommenders
    ```
3. Run the generate conda file script and create a conda environment:   
    ```
    cd Recommenders
    ./scripts/generate_conda_file.sh
    conda env create -n reco -f conda_bare.yaml  
    ```
4. Activate the conda environment and register it with Jupyter:
    ```
    conda activate reco
    python -m ipykernel install --user --name reco --display-name "Python (reco)"
    ```
5. Start the Jupyter notebook server
    ```
    cd notebooks
    jupyter notebook
    ```
5. Run the [SAR Python CPU Movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) notebook under the 00_quick_start folder. Make sure to change the kernel to "Python (reco)".

## Notebooks Overview

- The [Quick-Start Notebooks](notebooks/00_quick_start) detail how you can quickly get up and run with state-of-the-art algorithms such as the Smart Adaptive Recommendation ([SAR](https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md)) algorithm and ALS algorithm. 

- The [Data Preparation Notebook](notebooks/01_prepare_data) shows how to prepare and split data properly for recommendation systems.

- The [Modeling Notebooks](notebooks/02_model) provide a deep dive into implementations of different recommender algorithms.

- The [Evaluation Notebook](notebooks/03_evaluate) shows how to evaluate recommender algorithms for different ranking and rating metrics.

- The [Operationalizion Notebook](notebooks/04_operationalize) demonstrates how to deploy models in production systems.

## Benchmarks

Below we show benchmark algorithm performance on the Movielens 1M dataset. Data is randomly split into train/test sets at at 75/25 ratio. A recommendation model is trained using each of the below collaborative filtering algorithms. We utilize empirical parameter values reported in literature that generated optimal results as reported [here](http://mymedialite.net/examples/datasets.html). For ranking metrics we use k = 10 (top 10 results). We benchmark on a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 K80 GPU). Spark ALS is run in local standalone mode. 

A [benchmark notebook](notebooks/03_evaluate/benchmark_notebook.ipynb) is provided to check out results on larger datasets, and/or reproduce the results under a custom settings. 

**Benchmark results**

| Algo | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Explained Variance | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| ALS | 0.002020 | 0.024313 | 0.030677 | 0.009649 | 0.860502 | 0.680608 | 0.406014 | 0.411603 | 
| SAR | 0.064013 | 0.308012 | 0.277215 | 0.109292 | NaN | NaN | NaN | NaN | 
| SVD | 0.010915 | 0.102398 | 0.092996 | 0.025362 | 0.888991 | 0.696781 | 0.364178 | 0.364178 | 

## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** |  master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792)  | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | 
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4805)|

**NOTE** - the tests are executed every night, we use `pytest` for testing python utilities in [reco_utils](reco_utils) and [notebooks](notebooks).

