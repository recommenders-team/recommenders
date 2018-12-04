# Operationalize

In this directory, notebooks are provided to demonstrate how recommendation systems developed in a heterogeneous environment (e.g., Spark, GPU, etc.) can be operationalized.


## Workflow
The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](/reco_workflow.png)

A few Azure services are recommended for scalable data storage ([Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)), model development ([Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (DSVM), [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service/)), and model opertionalization ([Azure Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/) (AKS)). 

![architecture](/reco-arch.png)
