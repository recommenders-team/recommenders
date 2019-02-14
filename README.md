# Recommenders

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learnings on five key tasks: 
- [Prepare Data](notebooks/01_prepare_data/README.md): Preparing and loading data for each recommender algorithm
- [Model](notebooks/02_model/README.md): Building models using various recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)), Singular Value Decomposition ([SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)), etc.
- [Evaluate](notebooks/03_evaluate/README.md): Evaluating algorithms with offline metrics
- [Model Select and Optimize](notebooks/04_model_select_and_optimize): Tuning and optimizing hyperparameters for recommender models
- [Operationalize](notebooks/05_operationalize/README.md): Operationalizing models in a production environment on Azure

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

## Algorithms

The table below lists recommender algorithms available in the repository at the moment.

| Algorithm | Environment | Type | Description | 
| --- | --- | --- | --- |
| [Surprise/Singular Value Decomposition (SVD)](notebooks/00_quick_start/sar_movielens.ipynb) | CPU-only | Collaborative Filtering | Matrix factorization algorithm for predicting explicit rating feedback in small datasets | 
| [Alternating Least Squares (ALS)](notebooks/00_quick_start/als_movielens.ipynb) | Spark | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets, optimized with Spark for scalability and distributed computing capability |
| [Smart Adaptive Recommendations (SAR)](notebooks/00_quick_start/sar_movielens.ipynb) | CPU-only | Collaborative Filtering | Similarity-based algorithm for implicit feedback dataset |
| [Vowpal Wabbit Family (VW)](notebooks/02_model/vowpal_wabbit_deep_dive.ipynb) | CPU-only (train online) | Collaborative, Content-based Filtering | Fast online learning algorithms, great for scenarios where user features / context are constantly changing, like real-time bidding |
| [eXtreme Deep Factorization Machine (xDeepFM)](notebooks/00_quick_start/xdeepfm_synthetic.ipynb) | CPU / GPU | Hybrid | Deep learning based algorithm for implicit and explicit feedback with user/item features | 
| [Deep Knowledge-Aware Network (DKN)](notebooks/00_quick_start/dkn_synthetic.ipynb) | CPU / GPU | Content-based Filtering | Deep learning algorithm incorporating a knowledge graph and article embeddings to provide powerful news or article recommendations |  
| [Neural Collaborative Filtering (NCF)](notebooks/00_quick_start/ncf_movielens.ipynb) | CPU / GPU | Collaborative Filtering | Deep learning algorithm with enhanced performance for implicit feedback | 
| [Restricted Boltzmann Machines (RBM)](notebooks/00_quick_start/rbm_movielens.ipynb) | CPU / GPU | Collaborative Filtering | Neural network based algorithm to learn the underlying probability distribution for explicit or implicit feedback | 
| [FastAI Embedding Dot Bias (FAST)](notebooks/00_quick_start/fastai_movielens.ipynb)  | CPU / GPU | Collaborative Filtering | General purpose algorithm with embedding dot biases for users and items |

**Preliminary Comparison**

We provide a [comparison notebook](notebooks/03_evaluate/comparison.ipynb) to illustrate how different algorithms could be evaluated and compared. In this notebook, data (MovieLens 1M) is randomly split into train/test sets at a 75/25 ratio. A recommendation model is trained using each of the collaborative filtering algorithms below. We utilize empirical parameter values reported in literature [here](http://mymedialite.net/examples/datasets.html). For ranking metrics we use k = 10 (top 10 results). We run the comparison on a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 K80 GPU). Spark ALS is run in local standalone mode. 

| Algo | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Explained Variance | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| [ALS](notebooks/00_quick_start/als_movielens.ipynb) | 0.002020 | 0.024313 | 0.030677 | 0.009649 | 0.860502 | 0.680608 | 0.406014 | 0.411603 | 
| [SVD](notebooks/02_model/surprise_svd_deep_dive.ipynb) | 0.010915 | 0.102398 | 0.092996 | 0.025362 | 0.888991 | 0.696781 | 0.364178 | 0.364178 | 
| [FastAI](notebooks/00_quick_start/fastai_movielens.ipynb) | 0.023022 |0.168714 |0.154761 |0.050153 |0.887224 |0.705609 |0.371552 |0.374281 |


## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** |  master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792)  | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | 
| **Linux GPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu?branchName=master)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4997) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_staging?branchName=staging)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4998)|
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/Recommenders/nightly_spark_staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=5186)|

**NOTE** - the tests are executed every night, we use `pytest` for testing python utilities in [reco_utils](reco_utils) and `papermill` for the [notebooks](notebooks).

