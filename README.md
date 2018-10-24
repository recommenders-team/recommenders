| Build Type      | Status | Branch |
| ---             | ---    | ---       |
| **Linux CPU**   | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/staging_nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | master |
| **Linux CPU**   | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/staging_nightly?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | staging |


# Microsoft Recommenders

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learning to illustrate four key tasks: 
1. preparing and loading data for each recommender algorithm. 
2. Using different algorithms such as SAR, ALS, xDeepFFM, etc., for building recommender models. 
3. Evaluating algorithms with offline metrics. 
4. Operationalizing models in a production environment. The examples work across Python + cpu, Python + gpu, and PySpark environments, and contain guidance as to which algorithm to run in which environment based on scale and other requirements. 

Several utilities are provided in [reco_utils](/reco_utils) which will help accelerate experimenting with and building recommendation systems. These utility functions are used to load datasets (i.e., via Pandas DataFrames in python and via Spark DataFrames in Spark) in the manner expected by different algorithms, evaluate different model outputs, split training data, and perform other common tasks. Reference implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications. 

## Environment Setup
* Please see the [setup guide](SETUP.md).

## Notebooks Overview

- The [Quick-Start Notebooks](notebooks/00_quick_start/) detail how you can quickly get up and run with state-of-the-art algorithms such as the Smart Adaptive Recommendation (SAR) algorithm. 
- The [Data Notebooks](notebooks/01_data) detail how to prepare and split data properly for recommendation systems
- The [Modeling Notebooks](notebooks/02_modeling) deep dive into implemetnations of different recommender algorithms
- The [Evaluate Notebooks](notebooks/03_evaluate) discuss how to evaluate recommender algorithms for different ranking and rating metrics
- The [Operationalize Notebooks](notebooks/04_operationalize) discuss how to deploy models in production systems

| Notebook | Description | 
| --- | --- | 
| [sar_python_cpu_movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a Python+CPU environment
| [sar_pyspark_movielens](notebooks/00_quick_start/sar_pyspark_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a PySpark environment
| [data_split](notebooks/01_data/data_split.ipynb) | Details on splitting data (randomly, chronologically, etc)
| [sar_deep_dive](notebooks/02_modeling/sar_deep_dive.ipynb) | Deep dive on the Smart Adaptive Rankings algorithm and implementation


## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


