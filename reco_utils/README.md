# Recommender Utilities

This package contains functions to simplify common tasks used when developing and evaluating recommender systems. A short description of the sub-modules is provided below. For more details about what functions are available and how to use them, please review the doc-strings provided with the code.

See the [online documentation](https://readthedocs.org/projects/microsoft-recommenders/).

# Installation

## Pre-requisites
Some dependencies require compilation during pip installation, on linux this can be supported by adding build-essential dependencies:
```bash
sudo apt-get install -y build-essential
```

On Windows you will need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Install core utilities, cpu-based algorithms, and dependencies
```bash
pip install recommenders
```

## Optional Dependencies

By default recommenders does not install all dependencies used throughout the code or the notebook examples in this repo. Instead we require a bare minimum set of dependencies needed to execute functionality in the recommenders package (excluding Spark and GPU functionality). We also allow the user to specify which groups of dependencies are needed at installation time (or later if updating the pip installation). The following groups are provided:

- examples: dependencies needed to run [example notebooks](https://github.com/microsoft/recommenders/tree/main/examples)
- gpu: dependencies to enable GPU functionality (PyTorch & TensorFlow)
- spark: dependencies to enable Apache Spark functionality used in dataset, splitting, evaluation
- xlearn: xLearn package (on some platforms it requires pre-installation of cmake)
- all: all of the above dependencies
- experimental: current experimental dependencies that are being evaluated (e.g. libraries that require advanced build requirements or might conflict with libraries from other options)

These groups can be installed alone or in combination:
```bash
# install recommenders with core requirements and support for all recommender algorithms
pip install recommenders[examples]

# add support for running example notebooks and gpu functionality
pip install recommenders[examples,gpu]
```

## GPU Support

You will need CUDA Toolkit v10.0 and CuDNN >= 7.6 to enable both Tensorflow and PyTorch to use the GPU. For example, if you are using a conda enviroment, this can be installed with
```bash
conda install cudatoolkit=10.0 "cudnn>=7.6"
```

For manual installation of the necessary requirements see [TensorFlow](https://www.tensorflow.org/install/gpu#software_requirements) and [PyTorch](https://pytorch.org/get-started/locally/) installation pages.

When installing with GPU support you will need to point to the PyTorch index to ensure you are downloading a version of PyTorch compiled with CUDA support. This can be done using the --find-links or -f option below.

`pip install recommenders[gpu] -f https://download.pytorch.org/whl/cu100/torch_stable.html`

## Experimental dependencies

We are currently evaluating inclusion of the following dependencies:

 - vowpalwabbit: current examples show how to use vowpal wabbit after it has been installed on the command line; using the [PyPI package](https://pypi.org/project/vowpalwabbit/) with the scikit-learn interface will facilitate easier integration into python environments

# Contents

## [Common](common)

This submodule contains high-level utilities for defining constants used in most algorithms as well as helper functions for managing aspects of different frameworks: GPU, Spark, Jupyter notebook.

## [Dataset](dataset)

Dataset includes helper functions for pulling different datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

### Data Loading

There are dataloaders for several datasets. For example, the movielens module will allow you to load a dataframe in pandas or spark formats from the MovieLens dataset, with sizes of 100k, 1M, 10M, or 20M to test algorithms and evaluate performance benchmarks.

```python
df = movielens.load_pandas_df(size="100k")
```

### Splitting Techniques

Currently three methods are available for splitting datasets. All of them support splitting by user or item and filtering out minimal samples (for instance users that have not rated enough items, or items that have not been rated by enough users).

- Random: this is the basic approach where entries are randomly assigned to each group based on the ratio desired
- Chronological: this uses provided timestamps to order the data and selects a cut-off time that will split the desired ratio of data to train before that time and test after that time
- Stratified: this is similar to random sampling, but the splits are stratified, for example if the datasets are split by user, the splitting approach will attempt to maintain the same set of items used in both training and test splits. The converse is true if splitting by item.

## [Evaluation](evaluation)

The evaluation submodule includes functionality for calculating common recommender metrics directly in python or in a Spark environment using pyspark.

Currently available metrics include:

- Root Mean Squared Error
- Mean Absolute Error
- R<sup>2</sup>
- Explained Variance
- Precision at K
- Recall at K
- Normalized Discounted Cumulative Gain at K
- Mean Average Precision at K
- Area Under Curve
- Logistic Loss

## [Recommender](recommender)

The recommender submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new recommender system approaches. A description of all the algorithms can be found on [this table](../README.md#algorithms). Next a list of the algorithm utilities:

* Cornac
* DeepRec
  *  Convolutional Sequence Embedding Recommendation (CASER)
  *  Deep Knowledge-Aware Network (DKN)
  *  Extreme Deep Factorization Machine (xDeepFM)
  *  GRU4Rec
  *  LightGCN
  *  Next Item Recommendation (NextItNet)
  *  Short-term and Long-term Preference Integrated Recommender (SLi-Rec)
  *  Multi-Interest-Aware Sequential User Modeling (SUM)
* FastAI
* GeoIMC
* LightFM
* LightGBM
* NCF
* NewsRec
  * Neural Recommendation with Long- and Short-term User Representations (LSTUR)
  * Neural Recommendation with Attentive Multi-View Learning (NAML)
  * Neural Recommendation with Personalized Attention (NPA)
  * Neural Recommendation with Multi-Head Self-Attention (NRMS)
* Restricted Boltzmann Machines (RBM)
* Riemannian Low-rank Matrix Completion (RLRMC)
* Simple Algorithm for Recommendation (SAR)
* Surprise
* Term Frequency - Inverse Document Frequency (TF-IDF)
* Variational Autoencoders (VAE)
  * Multinomial
  * Standard
* Vowpal Wabbit (VW)
* Wide and Deep
* xLearn
  * Factorization Machine (FM)
  * Field-Aware FM (FFM)

## [Tuning](tuning)

This submodule contains utilities for performing hyperparameter tuning.
