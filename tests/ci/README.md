# Testing strategy

Here we describe the testing strategy for the Recommenders repository.

## Testing pipelines

### Azure Machine Learning service testing pipeline

The Python files to enable the AzureML tests are located in [azureml_tests](azureml_tests).

The GitHub workflows for testing pipelines that run within an Azure Machine Learning service workspace are located in [recommenders/.github/workflows/](../../.github/workflows/).

### Azure Data Science Virtual Machine testing pipelines (deprecated)

These are the testing pipelines that run on either a Linux or Windows DSVM agent machine. The yaml files can be found in [azure_pipeline_tests](azure_pipeline_test).