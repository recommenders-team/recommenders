# Operationalize

In this directory, notebooks are provided to demonstrate how to 

| Notebook | Description | 
| --- | --- | 
| [hypertune_spark_deep_dive](hypertune_spark_deep_dive.ipynb) | Step by step tutorials on how to fine tune hyperparameters for Spark based recommender model (illustrated by Spark ALS) with [Spark native construct](https://spark.apache.org/docs/2.3.1/ml-tuning.html) and [`hyperopt` package](http://hyperopt.github.io/hyperopt/).


## Workflow
The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](/notebooks/05_operationalize/reco_workflow.png)

## Reference Architecture
A few Azure services are recommended for scalable data storage ([Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)), model development ([Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (DSVM), [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/)), and model operationalization ([Azure Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/) (AKS)). 

![architecture](/notebooks/05_operationalize/reco-arch.png)
