# Model Select and Optimize

In this directory, notebooks are provided to demonstrate how to tune and optimize hyperparameters of recommender algorithms with the utility functions ([reco_utils](../../reco_utils)) provided in the repository. 

| Notebook | Description | 
| --- | --- | 
| [hypertune_spark_deep_dive](hypertune_spark_deep_dive.ipynb) | Step by step tutorials on how to fine tune hyperparameters for Spark based recommender model (illustrated by Spark ALS) with [Spark native construct](https://spark.apache.org/docs/2.3.1/ml-tuning.html) and [`hyperopt` package](http://hyperopt.github.io/hyperopt/).
| [azureml_hyperdrive_tuning_svd.ipynb](azureml_hyperdrive_tuning_svd.ipynb) | Quickstart tutorial on utilizing [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) for hyperparameter tuning of the matrix factorization method SVD from [Surprise library](https://surprise.readthedocs.io/en/stable/).
