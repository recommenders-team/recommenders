# Testing strategy

## Naming convention of pipeline yaml files

We use underscore connected strings in a pipeline yaml file name to indicate the environment, branch, framework, etc. of a testing pipeline. The naming convention follows the pattern as below

```
<compute>_<test>_<os>_<environment>_<branch>.yaml
```

For example, if a unit test for Spark utility functions of master branch is run on an Linux Azure Data Science Virtual Machine, it will be named as 
```
dsvm_unit_linux_pyspark_master.yaml
```