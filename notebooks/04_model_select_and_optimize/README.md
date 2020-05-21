# Model Select and Optimize

In this directory, notebooks are provided to demonstrate how to tune and optimize hyperparameters of recommender algorithms with the utility functions ([reco_utils](../../reco_utils)) provided in the repository. 

| Notebook | Description | 
| --- | --- | 
| [tuning_spark_als](tuning_spark_als.ipynb) | Step by step tutorials on how to fine tune hyperparameters for Spark based recommender model (illustrated by Spark ALS) with [Spark native construct](https://spark.apache.org/docs/2.3.1/ml-tuning.html) and [`hyperopt` package](http://hyperopt.github.io/hyperopt/).
| [azureml_hyperdrive_wide_and_deep](azureml_hyperdrive_wide_and_deep.ipynb) | Quickstart tutorial on utilizing [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) for hyperparameter tuning of wide-and-deep model.
| [azureml_hyperdrive_surprise_svd](azureml_hyperdrive_surprise_svd.ipynb) | Quickstart tutorial on utilizing [Azure Machine Learning service](https://azure.microsoft.com/en-us/services/machine-learning-service/) for hyperparameter tuning of the matrix factorization method SVD from [Surprise library](https://surprise.readthedocs.io/en/stable/).
| [nni_surprise_svd](nni_surprise_svd.ipynb) | Quickstart tutorial on utilizing the [Neural Network Intelligence toolkit](https://github.com/Microsoft/nni) for hyperparameter tuning of the matrix factorization method SVD from [Surprise library](https://surprise.readthedocs.io/en/stable/).
| [nni_ncf](nni_ncf.ipynb) | Quickstart tutorial on utilizing the [Neural Network Intelligence toolkit](https://github.com/Microsoft/nni) as a tool to tune the [NCF model](../02_model/ncf_deep_dive.ipynb) and [SVD model](../02_model/surprise_svd_deep_dive.ipynb) and compare their performance against one another

### Prerequisites
To run the examples running on the Azure Machine Learning service, the [`azureml-sdk`](https://pypi.org/project/azureml-sdk/) is required. The AzureML Python SDK is already installed after setting up the conda environments from this repository (see [SETUP.md](../../SETUP.md)). 

More info about setting up an AzureML environment can be found at [this link](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-environment).

### AzureML Workspace Configuration
AzureML workspace is the foundational block in the cloud that you use to experiment, train, and deploy machine learning models. We 
1. set up a workspace from Azure portal and 
2. create a config file manually. 

The instructions here are based on AzureML documents about [Quickstart with Azure portal](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-get-started) and [Quickstart with Python SDK](https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python) where you can find more details with screenshots about the setup process.
  
#### Create a workspace
1. Sign in to the [Azure portal](https://portal.azure.com) by using the credentials for the Azure subscription you use.
2. Select **Create a resource** menu, search for **Machine Learning service workspace** select **Create** button.
3. In the **ML service workspace** pane, configure your workspace by entering the *workspace name* and *resource group* (or **create new** resource group if you don't have one already), and select **Create**. It can take a few moments to create the workspace.
  
#### Make a configuration file
To configure this notebook to communicate with your workspace, type in your Azure subscription id, the resource group name and workspace name to <subscription-id>, <resource-group>, <workspace-name> in the cell below. Alternatively, you can create a *./aml_config/config.json* file with the following contents:
```
{
    "subscription_id": "<subscription-id>",
    "resource_group": "<resource-group>",
    "workspace_name": "<workspace-name>"
}
```

### NNI Configuration
The NNI command `nnictl` comes installed with the conda environment. 
In order to use the SMAC tuner, it has to be installed first with the following command 
`nnictl package install --name=SMAC`.
