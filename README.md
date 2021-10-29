# Recommenders

[![Documentation Status](https://readthedocs.org/projects/microsoft-recommenders/badge/?version=latest)](https://microsoft-recommenders.readthedocs.io/en/latest/?badge=latest)

## What's New 
### (September 27, 2021)

We have a new release [Recommenders 0.7.0](https://github.com/microsoft/recommenders/releases/tag/0.7.0)!
We have changed the names of the folders which contain the source code, so that they are more informative. This implies that you will need to change any import statements that reference the recommenders package. Specifically, the folder `reco_utils` has been renamed to `recommenders` and its subfolders have been renamed according to [issue 1390](https://github.com/microsoft/recommenders/issues/1390).  

The previous release ([0.6.0](https://github.com/microsoft/recommenders/releases/tag/0.6.0)) is compatible with the old style of naming of modules. 

The recommenders package now supports three types of environments: [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/index.html#) and [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment) with Python versions 3.6 and 3.7.

We have also added new evaluation metrics: _novelty, serendipity, diversity and coverage_ (see the [evalution notebooks](examples/03_evaluate/README.md)).

Code coverage reports are now generated for every PR, using [Codecov](https://about.codecov.io/).

Starting with release 0.6.0, Recommenders has been available on PyPI and can be installed using pip! 

Here you can find the PyPi page: https://pypi.org/project/recommenders/

Here you can find the package documentation: https://microsoft-recommenders.readthedocs.io/en/latest/


## Introduction

This repository contains examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learnings on five key tasks:

- [Prepare Data](examples/01_prepare_data): Preparing and loading data for each recommender algorithm
- [Model](examples/00_quick_start): Building models using various classical and deep learning recommender algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or eXtreme Deep Factorization Machines ([xDeepFM](https://arxiv.org/abs/1803.05170)).
- [Evaluate](examples/03_evaluate): Evaluating algorithms with offline metrics
- [Model Select and Optimize](examples/04_model_select_and_optimize): Tuning and optimizing hyperparameters for recommender models
- [Operationalize](examples/05_operationalize): Operationalizing models in a production environment on Azure

Several utilities are provided in [recommenders](recommenders) to support common tasks such as loading datasets in the format expected by different algorithms, evaluating model outputs, and splitting training/test data. Implementations of several state-of-the-art algorithms are included for self-study and customization in your own applications. See the [recommenders documentation](https://readthedocs.org/projects/microsoft-recommenders/).

For a more detailed overview of the repository, please see the documents on the [wiki page](https://github.com/microsoft/recommenders/wiki/Documents-and-Presentations).

## Getting Started

Please see the [setup guide](SETUP.md) for more details on setting up your machine locally, on a [data science virtual machine (DSVM)](https://azure.microsoft.com/en-gb/services/virtual-machines/data-science-virtual-machines/) or on [Azure Databricks](SETUP.md#setup-guide-for-azure-databricks).

The installation of the recommenders package has been tested with 
- Python versions 3.6, 3.7 and [venv](https://docs.python.org/3/library/venv.html), [virtualenv](https://virtualenv.pypa.io/en/latest/index.html#) or [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment) 

and currently does not support version 3.8 and above. It is recommended to install the package and its dependencies inside a clean environment (such as [conda](https://docs.conda.io/projects/conda/en/latest/glossary.html?highlight=environment#conda-environment), [venv](https://docs.python.org/3/library/venv.html) or [virtualenv](https://virtualenv.pypa.io/en/latest/index.html#)).

To set up on your local machine:

To install core utilities, CPU-based algorithms, and dependencies:

1. Ensure software required for compilation and Python libraries is installed. On Linux this can be supported by adding:
```bash
sudo apt-get install -y build-essential libpython<version>
``` 
where `<version>` should be `3.6` or `3.7` as appropriate.

On Windows you will need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
  
2. Create a conda or virtual environment. See the [setup guide](SETUP.md) for more details.

3. Within the created environment, install the package from [PyPI](https://pypi.org):

```bash
pip install --upgrade pip
pip install --upgrade setuptools
pip install recommenders[examples]
```

4. Register your (conda or virtual) environment with Jupyter:

```bash
python -m ipykernel install --user --name my_environment_name --display-name "Python (reco)"
```

5. Start the Jupyter notebook server

```bash
jupyter notebook
```

6. Run the [SAR Python CPU MovieLens](examples/00_quick_start/sar_movielens.ipynb) notebook under the `00_quick_start` folder. Make sure to change the kernel to "Python (reco)".

For additional options to install the package (support for GPU, Spark etc.) see [this guide](recommenders/README.md).

**NOTE** - The [Alternating Least Squares (ALS)](examples/00_quick_start/als_movielens.ipynb) notebooks require a PySpark environment to run. Please follow the steps in the [setup guide](SETUP.md#dependencies-setup) to run these notebooks in a PySpark environment. For the deep learning algorithms, it is recommended to use a GPU machine and to follow the steps in the [setup guide](SETUP.md#dependencies-setup) to set up Nvidia libraries.

**NOTE for DSVM Users** - Please follow the steps in the [Dependencies setup - Set PySpark environment variables on Linux or MacOS](SETUP.md#dependencies-setup) and [Troubleshooting for the DSVM](SETUP.md#troubleshooting-for-the-dsvm) sections if you encounter any issue.

**DOCKER** - Another easy way to try the recommenders repository and get started quickly is to build [docker images](tools/docker/README.md) suitable for different environments. 

## Algorithms

The table below lists the recommender algorithms currently available in the repository. Notebooks are linked under the Environment column when different implementations are available.

| Algorithm | Type | Description | Example |
|-----------|------|-------------|---------|
| Alternating Least Squares (ALS) | Collaborative Filtering | Matrix factorization algorithm for explicit or implicit feedback in large datasets, optimized for scalability and distributed computing capability. It works in the PySpark environment. | [Quick start](examples/00_quick_start/als_movielens.ipynb) / [Deep dive](examples/02_model_collaborative_filtering/als_deep_dive.ipynb) |
| Attentive Asynchronous Singular Value Decomposition (A2SVD)<sup>*</sup> | Collaborative Filtering | Sequential-based algorithm that aims to capture both long and short-term user preferences using attention mechanism. It works in the CPU/GPU environment. | [Quick start](examples/00_quick_start/sequential_recsys_amazondataset.ipynb) |
| Cornac/Bayesian Personalized Ranking (BPR) | Collaborative Filtering | Matrix factorization algorithm for predicting item ranking with implicit feedback. It works in the CPU environment. | [Deep dive](examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb) |
| Cornac/Bilateral Variational Autoencoder (BiVAE) | Collaborative Filtering | Generative model for dyadic data (e.g., user-item interactions). It works in the CPU/GPU enviroment. | [Deep dive](examples/02_model_collaborative_filtering/cornac_bivae_deep_dive.ipynb) |
| Convolutional Sequence Embedding Recommendation (Caser) | Collaborative Filtering | Algorithm based on convolutions that aim to capture both user’s general preferences and sequential patterns. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/sequential_recsys_amazondataset.ipynb) |
| Deep Knowledge-Aware Network (DKN)<sup>*</sup> | Content-Based Filtering | Deep learning algorithm incorporating a knowledge graph and article embeddings to provide powerful news or article recommendations. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/dkn_MIND.ipynb) / [Deep dive](examples/02_model_content_based_filtering/dkn_deep_dive.ipynb) |
| Extreme Deep Factorization Machine (xDeepFM)<sup>*</sup> | Hybrid | Deep learning based algorithm for implicit and explicit feedback with user/item features. It works in the CPU/GPU environment. | [Quick start](examples/00_quick_start/xdeepfm_criteo.ipynb) |
| FastAI Embedding Dot Bias (FAST) | Collaborative Filtering | General purpose algorithm with embeddings and biases for users and items. It works in the CPU/GPU environment. | [Quick start](examples/00_quick_start/fastai_movielens.ipynb) |
| LightFM/Hybrid Matrix Factorization | Hybrid | Hybrid matrix factorization algorithm for both implicit and explicit feedbacks. It works in the CPU environment. | [Quick start](examples/02_model_hybrid/lightfm_deep_dive.ipynb) |
| LightGBM/Gradient Boosting Tree<sup>*</sup> | Content-Based Filtering | Gradient Boosting Tree algorithm for fast training and low memory usage in content-based problems. It works in CPU/GPU/PySpark environment. | [Quick start in CPU](examples/00_quick_start/lightgbm_tinycriteo.ipynb) / [Deep dive in PySpark](examples/02_model_content_based_filtering/mmlspark_lightgbm_criteo.ipynb) |
| LightGCN | Collaborative Filtering | Deep learning algorithm which simplifies the design of GCN for predicting implicit feedback. It works in the CPU/GPU enviroment. | [Deep dive](examples/02_model_collaborative_filtering/lightgcn_deep_dive.ipynb) |
| GeoIMC<sup>*</sup> | Hybrid | Matrix completion algorithm that has into account user and item features using Riemannian conjugate gradients optimization and following a geometric approach. It works in the CPU enviroment. | [Quick start](examples/00_quick_start/geoimc_movielens.ipynb) |
| GRU4Rec | Collaborative Filtering | Sequential-based algorithm that aims to capture both long and short-term user preferences using recurrent neural networks. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/sequential_recsys_amazondataset.ipynb) |
| Multinomial VAE | Collaborative Filtering | Generative model for predicting user/item interactions. It works in the CPU/GPU enviroment. | [Deep dive](examples/02_model_collaborative_filtering/multi_vae_deep_dive.ipynb) |
| Neural Recommendation with Long- and Short-term User Representations (LSTUR)<sup>*</sup> | Content-Based Filtering | Neural recommendation algorithm for recommending news articles with long- and short-term user interest modeling. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/lstur_MIND.ipynb) |
| Neural Recommendation with Attentive Multi-View Learning (NAML)<sup>*</sup> | Content-Based Filtering | Neural recommendation algorithm for recommending news articles with attentive multi-view learning. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/naml_MIND.ipynb) |
| Neural Collaborative Filtering (NCF) | Collaborative Filtering | Deep learning algorithm with enhanced performance for user/item implicit feedback. It works in the CPU/GPU enviroment.| [Quick start](examples/00_quick_start/ncf_movielens.ipynb) |
| Neural Recommendation with Personalized Attention (NPA)<sup>*</sup> | Content-Based Filtering | Neural recommendation algorithm for recommending news articles with personalized attention network. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/npa_MIND.ipynb) |
| Neural Recommendation with Multi-Head Self-Attention (NRMS)<sup>*</sup> | Content-Based Filtering | Neural recommendation algorithm for recommending news articles with multi-head self-attention. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/nrms_MIND.ipynb) |
| Next Item Recommendation (NextItNet) | Collaborative Filtering | Algorithm based on dilated convolutions and residual network that aims to capture sequential patterns. It considers both user/item interactions and features.  It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/sequential_recsys_amazondataset.ipynb) |
| Restricted Boltzmann Machines (RBM) | Collaborative Filtering | Neural network based algorithm for learning the underlying probability distribution for explicit or implicit user/item feedback. It works in the CPU/GPU enviroment. | [Quick start](examples/00_quick_start/rbm_movielens.ipynb) / [Deep dive](examples/02_model_collaborative_filtering/rbm_deep_dive.ipynb) |
| Riemannian Low-rank Matrix Completion (RLRMC)<sup>*</sup> | Collaborative Filtering | Matrix factorization algorithm using Riemannian conjugate gradients optimization with small memory consumption to predice user/item interactions. It works in the CPU enviroment. | [Quick start](examples/00_quick_start/rlrmc_movielens.ipynb) |
| Simple Algorithm for Recommendation (SAR)<sup>*</sup> | Collaborative Filtering | Similarity-based algorithm for implicit user/item feedback.  It works in the CPU environment. | [Quick start](examples/00_quick_start/sar_movielens.ipynb) / [Deep dive](examples/02_model_collaborative_filtering/sar_deep_dive.ipynb) |
| Short-term and Long-term Preference Integrated Recommender (SLi-Rec)<sup>*</sup> | Collaborative Filtering | Sequential-based algorithm that aims to capture both long and short-term user preferences using attention mechanism, a time-aware controller and a content-aware controller. It works in the CPU/GPU environment. | [Quick start](examples/00_quick_start/sequential_recsys_amazondataset.ipynb) |
| Multi-Interest-Aware Sequential User Modeling (SUM)<sup>*</sup> | Collaborative Filtering | An enhanced memory network-based sequential user model which aims to capture users' multiple interests. It works in the CPU/GPU environment. | [Quick start](examples/00_quick_start/sequential_recsys_amazondataset.ipynb) |
| Standard VAE | Collaborative Filtering | Generative Model for predicting user/item interactions.  It works in the CPU/GPU environment. | [Deep dive](examples/02_model_collaborative_filtering/standard_vae_deep_dive.ipynb) |
| Surprise/Singular Value Decomposition (SVD) | Collaborative Filtering | Matrix factorization algorithm for predicting explicit rating feedback in small datasets. It works in the CPU/GPU environment. | [Deep dive](examples/02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb) |
| Term Frequency - Inverse Document Frequency (TF-IDF) | Content-Based Filtering | Simple similarity-based algorithm for content-based recommendations with text datasets. It works in the CPU environment. | [Quick staert](examples/00_quick_start/tfidf_covid.ipynb) |
| Vowpal Wabbit (VW)<sup>*</sup> | Content-Based Filtering | Fast online learning algorithms, great for scenarios where user features / context are constantly changing. It uses the CPU for online learning. | [Deep dive](examples/02_model_content_based_filtering/vowpal_wabbit_deep_dive.ipynb) |
| Wide and Deep | Hybrid | Deep learning algorithm that can memorize feature interactions and generalize user features. It works in the CPU/GPU environment. | [Quick start](examples/00_quick_start/wide_deep_movielens.ipynb) |
| xLearn/Factorization Machine (FM) & Field-Aware FM (FFM) | Hybrid | Quick and memory efficient algorithm to predict labels with user/item features. It works in the CPU/GPU environment. | [Deep dive](examples/02_model_hybrid/fm_deep_dive.ipynb) |

**NOTE**: <sup>*</sup> indicates algorithms invented/contributed by Microsoft.

Independent or incubating algorithms and utilities are candidates for the [contrib](contrib) folder. This will house contributions which may not easily fit into the core repository or need time to refactor or mature the code and add necessary tests.

| Algorithm | Type | Description | Example |
|-----------|------|-------------|---------|
| SARplus <sup>*</sup> | Collaborative Filtering | Optimized implementation of SAR for Spark |  [Quick start](contrib/sarplus/README.md) |

### Algorithm Comparison

We provide a [benchmark notebook](examples/06_benchmarks/movielens.ipynb) to illustrate how different algorithms could be evaluated and compared. In this notebook, the MovieLens dataset is split into training/test sets at a 75/25 ratio using a stratified split. A recommendation model is trained using each of the collaborative filtering algorithms below. We utilize empirical parameter values reported in literature [here](http://mymedialite.net/examples/datasets.html). For ranking metrics we use `k=10` (top 10 recommended items). We run the comparison on a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 P100 GPU). Spark ALS is run in local standalone mode. In this table we show the results on Movielens 100k, running the algorithms for 15 epochs.

| Algo | MAP | nDCG@k | Precision@k | Recall@k | RMSE | MAE | R<sup>2</sup> | Explained Variance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [ALS](examples/00_quick_start/als_movielens.ipynb) | 0.004732 |	0.044239 |	0.048462 |	0.017796 | 0.965038 |	0.753001 |	0.255647 |	0.251648 |
| [BiVAE](examples/02_model_collaborative_filtering/cornac_bivae_deep_dive.ipynb) | 0.146126	| 0.475077 |	0.411771 |	0.219145 | N/A |	N/A |	N/A |	N/A |
| [BPR](examples/02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb) | 0.132478	| 0.441997 |	0.388229 |	0.212522 | N/A |	N/A |	N/A |	N/A |
| [FastAI](examples/00_quick_start/fastai_movielens.ipynb) | 0.025503 |	0.147866 |	0.130329 |	0.053824 | 0.943084 |	0.744337 |	0.285308 |	0.287671 |
| [LightGCN](examples/02_model_collaborative_filtering/lightgcn_deep_dive.ipynb) | 0.088526 | 0.419846 | 0.379626 | 0.144336 | N/A | N/A | N/A | N/A |
| [NCF](examples/02_model_hybrid/ncf_deep_dive.ipynb) | 0.107720	| 0.396118 |	0.347296 |	0.180775 | N/A | N/A | N/A | N/A |
| [SAR](examples/00_quick_start/sar_movielens.ipynb) | 0.110591 |	0.382461 | 	0.330753 | 0.176385 | 1.253805 | 1.048484 |	-0.569363 |	0.030474 |
| [SVD](examples/02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb) | 0.012873	| 0.095930 |	0.091198 |	0.032783 | 0.938681 | 0.742690 | 0.291967 | 0.291971 |

## Code of Conduct

This project adheres to [Microsoft's Open Source Code of Conduct](CODE_OF_CONDUCT.md) in order to foster a welcoming and inspiring communtity for all.

## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).

## Build Status

These tests are the nightly builds, which compute the smoke and integration tests. `main` is our principal branch and `staging` is our development branch. We use `pytest` for testing python utilities in [recommenders](recommenders) and `papermill` for the [notebooks](examples). For more information about the testing pipelines, please see the [test documentation](tests/README.md).

### DSVM Build Status

The following tests run on a Linux DSVM daily. These machines run 24/7.

| Build Type | Branch | Status |  | Branch | Status |
| --- | --- | --- | --- | --- | --- |
| **Linux CPU** | main | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_cpu?branchName=main)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=162&branchName=main) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_cpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=162&branchName=staging) |
| **Linux GPU** | main | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_gpu?branchName=main)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=163&branchName=main) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_gpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=163&branchName=staging) |
| **Linux Spark** | main | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_pyspark?branchName=main)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=164&branchName=main) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/linux-tests/dsvm_nightly_linux_pyspark?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=164&branchName=staging) |
<!--
| **Windows CPU** | main | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_cpu?branchName=main)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=101&branchName=main) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_cpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=101&branchName=staging) |
| **Windows GPU** | main | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_gpu?branchName=main)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=102&branchName=main) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_gpu?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=102&branchName=staging) |
| **Windows Spark** | main | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_pyspark?branchName=main)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=103&branchName=main) | | staging | [![Build Status](https://dev.azure.com/best-practices/recommenders/_apis/build/status/windows-tests/dsvm_nightly_win_pyspark?branchName=staging)](https://dev.azure.com/best-practices/recommenders/_build/latest?definitionId=103&branchName=staging) |
-->

## Related projects

- [Microsoft AI Github](https://github.com/microsoft/ai): Find other Best Practice projects, and Azure AI design patterns in our central repository.
- [NLP best practices](https://github.com/microsoft/nlp-recipes): Best practices and examples on NLP.
- [Computer vision best practices](https://github.com/microsoft/computervision-recipes): Best practices and examples on computer vision.
- [Forecasting best practices](https://github.com/microsoft/forecasting): Best practices and examples on time series forecasting.

## Reference papers

- A. Argyriou, M. González-Fierro, and L. Zhang, "Microsoft Recommenders: Best Practices for Production-Ready Recommendation Systems", *WWW 2020: International World Wide Web Conference Taipei*, 2020. Available online: https://dl.acm.org/doi/abs/10.1145/3366424.3382692
- L. Zhang, T. Wu, X. Xie, A. Argyriou, M. González-Fierro and J. Lian, "Building Production-Ready Recommendation System at Scale", *ACM SIGKDD Conference on Knowledge Discovery and Data Mining 2019 (KDD 2019)*, 2019.
- S. Graham,  J.K. Min, T. Wu, "Microsoft recommenders: tools to accelerate developing recommender systems", *RecSys '19: Proceedings of the 13th ACM Conference on Recommender Systems*, 2019. Available online: https://dl.acm.org/doi/10.1145/3298689.3346967
