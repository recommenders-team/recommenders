# Testing strategy

## Naming convention of ADO pipeline yaml files

We use underscore connected strings in a pipeline yaml file name to indicate the environment, branch, framework, etc. of a testing pipeline. The naming convention follows the pattern as below

```
<compute>_<test>_<os>_<environment>.yml
```

For example, if a unit test for Spark utility functions is run on an Linux Azure Data Science Virtual Machine, it will be named as:

```
dsvm_unit_linux_pyspark.yml
```

## Testing pipelines

* Azure Data Science Virtual Machine testing pipelines

Testing pipelines that run on either a Linux or Windows DSVM agent machine.

* Azure Machine Learning service testing pipeline

GitHub workflows for testing pipelines that run within an Azure Machine Learning service workspace are located in `recommenders/.github/workflows/`.