# Testing strategy

## Naming convention of pipeline yaml files

We use underscore connected strings in a pipeline yaml file name to indicate the environment, branch, framework, etc. of a testing pipeline. The naming convention follows the pattern as below

```
<compute>_<test>_<os>_<environment>_<branch>.yml
```

For example, if a unit test for Spark utility functions of master branch is run on an Linux Azure Data Science Virtual Machine, it will be named as 
```
dsvm_unit_linux_pyspark_master.yml
```

If a test runs on both master and staging branches, its last field will be left empty.

## Testing pipelines

* Azure Data Science Virtual Machine testing pipelines

Testing pipelines that run on either a Linux or Windows DSVM agent machine. 

* Azure Machine Learning service testing pipeline

Testing pipelines that run within an Azure Machine Learning service workspace.

## Azure DevOps Templates 
Azure DevOps Templates have been used to reduce our duplicated code between repositories
For more information see [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/process/templates?view=azure-devops).

A Github Service Connection must also be created with the name "AI-GitHub" to use these templates, within each pipeline.
For more information see [here](https://docs.microsoft.com/en-us/azure/devops/pipelines/process/demands?view=azure-devops&tabs=yaml)

### [reco_config_conda_linux.yml@aitemplates](https://github.com/microsoft/AI/blob/master/.ci/steps/reco_config_conda_linux.yml)
This template is used to install a new conda env on a Linux Virtual Machine. The name of the conda env must be provided.

### [reco_conda_clean_linux.yml@aitemplates](https://github.com/microsoft/AI/blob/master/.ci/steps/reco_conda_clean_linux.yml)
This template is used to clean a Linux Virtual Machine after being used by a conda process. This should be used for a self-hosted linux agent.

### [reco_config_conda_win.yml@aitemplates](https://github.com/microsoft/AI/blob/master/.ci/steps/reco_conda_config_win.yml)
This template is used to install a new conda env on a Windows Virtual Machine. The name of the conda env must be provided.

### [reco_conda_clean_win.yml@aitemplates](https://github.com/microsoft/AI/blob/master/.ci/steps/reco_conda_clean_win.yml)
This template is used to clean a Windows Virtual Machine after being used by a conda process. This should be used for a self-hosted windows agent.