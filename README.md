| Build Type      | Status | Branch |
| ---             | ---    | ---       |
| **Linux CPU**   | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/staging_nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | master |
| **Linux CPU**   | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/staging_nightly?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | staging |


# Microsoft Recommenders

This repository provides examples and best practices for building recommender systems, provided as Jupyter notebooks. Our examples detail our learning to illustrate four key tasks: 1) preparing and loading data for each recommender, 2) utilizing different recommender algorithms such as SAR and DeepRec, 3) evaluating algorithms with rating or ranking metrics, and 4) operationalizing models in a production environment. Our examples work across python + cpu, python + gpu, and pyspark environments, and contain guidance as to which algorithm to run in which environment based on scale and other requirements. 

We provide several utilities (found in [reco_utils](/reco_utils)) which will help accelerate algorithm evaluation and selection. These utility functions are used to load datasets (ie via pandas in python and via spark dataframes in spark) in the manner expected by different algorithms, evaluate different model outputs, split training data, and perform other common tasks. We additionally provide reference implementations of several state-of-the-art algorithms that you can clone, modify, and reuse in your own applications. 

## Environment Setup
- Please see the [setup guide](SETUP.md).

## Notebooks Overview

- The [Quick-Start Notebooks](notebooks/00_quick_start/) that detail how you can quickly get up and running with state-of-the-art algorithms, including Smart Adaptive Rankings. 
- The [Data Notebooks](notebooks/01_data) detail how to prepare and split data properly for recommendation systems
- The [Modeling Notebooks](notebooks/02_modeling) deep dive into implemetnations of different recommender algorithms
- The [Evaluate Notebooks](notebooks/03_evaluate) discuss how to evaluate recommender algorithms for different ranking and rating metrics
- The [Operationalize Notebooks](notebooks/04_operationalize) discuss how to run your models in production systems

| Notebook | Description | 
| --- | --- | 
| [sar_python_cpu_movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a Python+CPU environment
| [sar_pyspark_movielens](notebooks/00_quick_start/sar_pyspark_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a PySpark environment
| [data_split](notebooks/01_data/data_split.ipynb) | Details on splitting data (randomly, chronologically, etc)
| [sar_deep_dive](notebooks/02_modeling/sar_deep_dive.ipynb) | Deep dive on the Smart Adaptive Rankings algorithm and implementation


## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


