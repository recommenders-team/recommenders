# Operationalize

In this directory, a notebook is provided to demonstrate how recommendation systems developed in a heterogeneous environment (e.g., Spark, GPU, etc.) can be operationalized.

| Notebook | Description | 
| --- | --- | 
| [als_movie_o16n](als_movie_o16n.ipynb) | End-to-end examples demonstrate how to build, evaluate, and deploy a Spark ALS based movie recommender with Azure services such as [Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction), and [Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/).


## Workflow
The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](https://recodatasets.blob.core.windows.net/images/reco_workflow.png)


## Reference Architecture
A few Azure services are recommended for scalable data storage ([Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)), model development ([Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (DSVM), [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/)), and model operationalization ([Azure Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/) (AKS)). 

![architecture](https://recodatasets.blob.core.windows.net/images/reco-arch.png)