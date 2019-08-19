# Operationalize

In this directory, a notebook is provided to demonstrate how recommendation systems developed in a heterogeneous environment (e.g., Spark, GPU, etc.) can be operationalized.

| Notebook | Description | 
| --- | --- | 
| [als_movie_o16n](als_movie_o16n.ipynb) | End-to-end examples demonstrate how to build, evaluate, and deploy a Spark ALS based movie recommender with Azure services such as [Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction), and [Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/).
| [aks_locust_load_test](aks_locust_load_test.ipynb) | Load test example for a recommendation system deployed on an AKS cluster | 
| [lightgbm_criteo_o16n](lightgbm_criteo_o16n.ipynb) | Content-based personalization deployment of a add click prediction scenario |
| [lgbm_webservice_poc](lgbm_webservice_poc.ipynb) | Example notebook for building LGBM based recommender model for use in the Recommenders Engine Example Layout app. |
| [sar_webservice_poc](sar_webservice_poc.ipynb) | Example notebook for building SAR based recommender model for use in the Recommenders Engine Example Layout app. |