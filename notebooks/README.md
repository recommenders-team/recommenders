# Recommender notebooks

This folder contains examples and best practices, written in Jupyter notebooks, for building recommendation systems.

## Summary

The following summarizes each directory of the best practice notebooks.

| Directory | Runs Local | Description |
| --- | --- | --- |
| [00_quick_start](./00_quick_start)| Yes | Quick start notebooks that demonstrate workflow of developing a recommender by using an algorithm in local environment|
| [01_prepare_data](./01_prepare_data) | Yes | Data preparation notebooks for each recommender algorithm|
| [02_model](./02_model) | Yes | Deep dive notebooks about model building by using various classical and deep learning recommender algorithms|
| [03_evaluate](./03_evaluate) | Yes | Notebooks that introduce different evaluation methods for recommenders|
| [04_model_select_and_optimize](04_model_select_and_optimize) | No, Run on Azure | Best practice notebooks for model tuning and selecting by using Azure Machine Learning Service and/or open source technologies|
| [05_operationalize](05_operationalize) | No, Run on Azure | Operationalization notebooks that illustrate an end-to-end pipeline by using a recommender algorithm for a certain real-world use case scenario|

## On-premise notebooks

The notebooks that do not require Azure can be run out-of-the-box on any Linux machine, where an environment is properly
set up by following the [instructions](../SETUP.md). 

**NOTE** Some of the notebooks may rely on heterogeneous computing instances
like a cluster of CPU machines with Spark framework installed or machines with GPU devices incorporated. It is therefore recommended
to run these notebooks on a [Data Science Virtual Machine for Linux (Ubuntu)](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-dsvm.linux-data-science-vm-ubuntu), where a single-node Spark and/or GPU device are pre-configured.

## Azure-enhanced notebooks

Azure products and services are used in certain notebooks to enhance the development-efficiency of recommender system in scale.
To successfully run these notebooks, the users **need an Azure subscription** or can [use Azure for free](https://azure.microsoft.com/en-us/free/).
The Azure products featured in the notebooks include:

* [Azure Machine Learning Service](https://azure.microsoft.com/en-us/services/machine-learning-service/) - it is used for model building (e.g., hyperparameter tuning), model serving, etc. Azure
Machine Learning Service is used intensively across various notebooks for the AI model development related tasks like
  * Hyperparameter tuning
  * Tracking and monitoring metrics to enhance the model creation process
  * Scaling up and out on Compute like DSVM and Azure Machine Learning Compute
  * Deploying a web service to AKS
  * Submitting pipelines
* [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) - Azure Data Science Virtual Machine is mainly used for a remote server where user
can easily configure the local as well as the cloud environment for running the example notebooks.
* [Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction) - Cosmos DB is used for preserving data. This is demonstrated in the operationalization example where
recommendation results generated from a model are preserved in Cosmos DB for real-time serving purpose.
* [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/) - Azure Databricks is mainly used for developing Spark based recommenders such as Spark ALS algorithm, in a distributed computing
environment.
* [Azure Kubernetes Service](https://azure.microsoft.com/en-us/services/kubernetes-service/) - Azure Kubernetes Service is used for serving a recommender model or consuming the results
generated from a recommender for a application service.

There may be other Azure service or products used in the notebooks. Introduction and/or reference of
those will be provided in the notebooks.

