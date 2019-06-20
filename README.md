# Recommenders

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learnings on five key tasks: 
- [Prepare Data](notebooks/01_prepare_data/README.md): Preparing and loading data for each recommender algorithm
- [Model](notebooks/02_model/README.md): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or eXtreme Deep Factorization Machines ([xDeepFM](https://arxiv.org/abs/1803.05170)).
- [Evaluate](notebooks/03_evaluate/README.md): Evaluating algorithms with offline metrics
- [Model Select and Optimize](notebooks/04_model_select_and_optimize): Tuning and optimizing hyperparameters for recommender models
- [Operationalize](notebooks/05_operationalize/README.md): Operationalizing models in a production environment on Azure

Several utilities are provided in [reco_utils](reco_utils) to support common tasks such as loading datasets in the format expected by different algorithms, evaluating model outputs, and splitting training/test data. Implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications.

## Getting Started
Please see the [setup guide](SETUP.md) for more details on setting up your machine locally, on Spark, or on [Azure Databricks](SETUP.md#setup-guide-for-azure-databricks). 

To setup on your local machine:
1. Install Anaconda with Python >= 3.6. [Miniconda](https://conda.io/miniconda.html) is a quick way to get started.
2. Clone the repository
    ```
    git clone https://github.com/Microsoft/Recommenders
    ```
3. Run the generate conda file script to create a conda environment:
   (This is for a basic python environment, see [SETUP.md](SETUP.md) for PySpark and GPU environment setup) 
    ```
    cd Recommenders
    python scripts/generate_conda_file.py
    conda env create -f reco_base.yaml  
    ```
4. Activate the conda environment and register it with Jupyter:
    ```
    conda activate reco_base
    python -m ipykernel install --user --name reco_base --display-name "Python (reco)"
    ```
5. Start the Jupyter notebook server
    ```
    cd notebooks
    jupyter notebook
    ```
6. Run the [SAR Python CPU MovieLens](notebooks/00_quick_start/sar_movielens.ipynb) notebook under the 00_quick_start folder. Make sure to change the kernel to "Python (reco)".

**NOTE** - The [Alternating Least Squares (ALS)](notebooks/00_quick_start/als_movielens.ipynb) notebooks require a PySpark environment to run. Please follow the steps in the [setup guide](SETUP.md#dependencies-setup) to run these notebooks in a PySpark environment.

## Install this repository via PIP
A [setup.py](reco_utils/setup.py) file is provied in order to simplify the installation of this utilities in this repo from the main directory. 

    pip install -e reco_utils

It is also possible to install directly from Github.

    pip install git+http://github.com/microsoft/recommenders/#egg=pkg&subdirectory=reco_utils

**NOTE** - The pip installation does not install any of the necessary package dependencies, it is expected that conda will be used as shown above to setup the environment for the utilities being used.


## Algorithms

The table below lists recommender algorithms currently available in the repository. Notebooks are linked under the Environment column when different implementations are available.

| Algorithm | Environment | Type | Description | 
| --- | --- | --- | --- |
| Alternating Least Squares (ALS) | [PySpark](notebooks/00_quick_start/als_movielens.ipynb) | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets, optimized by Spark MLLib for scalability and distributed computing capability | 
| Deep Knowledge-Aware Network (DKN)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/dkn_synthetic.ipynb) | Content-Based Filtering | Deep learning algorithm incorporating a knowledge graph and article embeddings to provide powerful news or article recommendations | 
| Extreme Deep Factorization Machine (xDeepFM)<sup>*</sup> | [Python CPU / Python GPU](notebooks/00_quick_start/xdeepfm_criteo.ipynb) | Hybrid | Deep learning based algorithm for implicit and explicit feedback with user/item features | 
| FastAI Embedding Dot Bias (FAST) | [Python CPU / Python GPU](notebooks/00_quick_start/fastai_movielens.ipynb) | Collaborative Filtering | General purpose algorithm with embeddings and biases for users and items |
| LightGBM/Gradient Boosting Tree<sup>*</sup> | [Python CPU](notebooks/00_quick_start/lightgbm_tinycriteo.ipynb) / [PySpark](notebooks/02_model/mmlspark_lightgbm_criteo.ipynb) | Content-Based Filtering | Gradient Boosting Tree algorithm for fast training and low memory usage in content-based problems |
| Neural Collaborative Filtering (NCF) | [Python CPU / Python GPU](notebooks/00_quick_start/ncf_movielens.ipynb) | Collaborative Filtering | Deep learning algorithm with enhanced performance for implicit feedback | 
| Restricted Boltzmann Machines (RBM) | [Python CPU / Python GPU](notebooks/00_quick_start/rbm_movielens.ipynb) | Collaborative Filtering | Neural network based algorithm for learning the underlying probability distribution for explicit or implicit feedback | 
| Riemannian Low-rank Matrix Completion (RLRMC)<sup>*</sup> | [Python CPU](notebooks/00_quick_start/rlrmc_movielens.ipynb) | Collaborative Filtering | Matrix factorization algorithm using Riemannian conjugate gradients optimization with small memory consumption. |
| Simple Algorithm for Recommendation (SAR)<sup>*</sup> | [Python CPU](notebooks/00_quick_start/sar_movielens.ipynb) | Collaborative Filtering | Similarity-based algorithm for implicit feedback dataset |
| Surprise/Singular Value Decomposition (SVD) | [Python CPU](notebooks/02_model/surprise_svd_deep_dive.ipynb) | Collaborative Filtering | Matrix factorization algorithm for predicting explicit rating feedback in datasets that are not very large | 
| Vowpal Wabbit Family (VW)<sup>*</sup> | [Python CPU (online training)](notebooks/02_model/vowpal_wabbit_deep_dive.ipynb) | Content-Based Filtering | Fast online learning algorithms, great for scenarios where user features / context are constantly changing |
| Wide and Deep | [Python CPU / Python GPU](notebooks/00_quick_start/wide_deep_movielens.ipynb) | Hybrid | Deep learning algorithm that can memorize feature interactions and generalize user features |


**NOTE**: <sup>*</sup> indicates algorithms invented/contributed by Microsoft.

**Preliminary Comparison**

We provide a [benchmark notebook](benchmarks/movielens.ipynb) to illustrate how different algorithms could be evaluated and compared. In this notebook, MovieLens dataset is splitted into training/test sets at a 75/25 ratio using a stratified split. A recommendation model is trained using each of the collaborative filtering algorithms below. We utilize empirical parameter values reported in literature [here](http://mymedialite.net/examples/datasets.html). For ranking metrics we use `k=10` (top 10 recommended items). We run the comparison on a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 P100 GPU). Spark ALS is run in local standalone mode. In this table we show the results on Movielens 100k, running the algorithms for 15 epochs.

| Algo | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Explained Variance | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| [ALS](notebooks/00_quick_start/als_movielens.ipynb) | 0.004732 |	0.044239 |	0.048462 |	0.017796 | 0.965038 |	0.753001 |	0.255647 |	0.251648 | 
| [SVD](notebooks/02_model/surprise_svd_deep_dive.ipynb) | 0.012873	| 0.095930 |	0.091198 |	0.032783 | 0.938681 |	0.742690	| 0.291967 |	0.291971 |
| [SAR](notebooks/00_quick_start/sar_movielens.ipynb) | 0.113028 |	0.388321 | 	0.333828 | 0.183179 | N/A |	N/A |	N/A |	N/A |
| [NCF](notebooks/02_model/ncf_deep_dive.ipynb) | 0.107720	| 0.396118 |	0.347296 |	0.180775 | N/A |	N/A |	N/A |	N/A |
| [FastAI](notebooks/00_quick_start/fastai_movielens.ipynb) | 0.025503 |	0.147866 |	0.130329 |	0.053824 | 0.943084 |	0.744337 |	0.285308 |	0.287671 |





## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status

| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) |
| **Linux GPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu?branchName=master)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4997) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_staging?branchName=staging)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4998) |
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/Recommenders/nightly_spark_staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=5186) |
| **Windows CPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_win?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=6743) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging_win?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=6752) |
| **Windows GPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_win?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=6756) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_staging_win?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=6761) |
| **Windows Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_win?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=6757) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_staging_win?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=6754) |

**NOTE** - these tests are the nightly builds, which compute the smoke and integration tests. Master is our main branch and staging is our development branch. We use `pytest` for testing python utilities in [reco_utils](reco_utils) and `papermill` for the [notebooks](notebooks). For more information about the testing pipelines, please see the [test documentation](tests/README.md).

