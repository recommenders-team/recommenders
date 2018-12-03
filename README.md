# Recommenders 

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learning to illustrate four key tasks: 
1. Preparing and loading data for each recommender algorithm. 
2. Using different algorithms such as Smart Adaptive Recommendation (SAR), Alternating Least Square (ALS), etc., for building recommender models. 
3. Evaluating algorithms with offline metrics. 
4. Operationalizing models in a production environment on Azure.

Several utilities are provided in [reco_utils](reco_utils) to do common tasks such as loading datasets in the manner expected by different algorithms, evaluate model outputs, and split training data. Reference implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications.

## Getting Started
Please see the [setup guide](SETUP.md) to setup including GPU or Spark dependencies or to [setup on Azure Databricks](/SETUP.md#setup-guide-for-azure-databricks). To setup on your local machine:
1. Install [Anaconda Python 3.6](https://conda.io/miniconda.html)
2. Run the generate conda file script and create a conda environment:   
    ```
    cd Recommenders
    ./scripts/generate_conda_file.sh
    conda env create -n reco -f conda_bare.yaml  
    ```
3. Activate the conda environment and register it with Jupyter:
    ```
    source activate my_env_name
    python -m ipykernel install --user --name my_env_name --display-name "Python (my_env_name)"
    ```
4. Run the [ALS Movielens Quickstart](notebooks/00_quick_start/als_pyspark_movielens.ipynb) notebook.

## Notebooks Overview

[Quick-Start Notebooks](notebooks/00_quick_start/) detail how you can quickly get up and run with state-of-the-art algorithms such as the Smart Adaptive Recommendation (SAR) algorithm. 

| Notebook | Description | 
| --- | --- | 
| [als_pyspark_movielens](notebooks/00_quick_start/als_pyspark_movielens.ipynb) | Utilizing the ALS algorithm to power movie ratings in a PySpark environment.
| [sar_python_cpu_movielens](notebooks/00_quick_start/sar_single_node_movielens.ipynb) | Utilizing the Smart Adaptive Recommendations (SAR) algorithm to power movie ratings in a Python+CPU environment.

[Data Notebooks](notebooks/01_data) detail how to prepare and split data properly for recommendation systems

| Notebook | Description | 
| --- | --- | 
| [data_split](notebooks/01_data/data_split.ipynb) | Details on splitting data (randomly, chronologically, etc).

The [Modeling Notebooks](notebooks/02_modeling) deep dive into implemetnations of different recommender algorithms

| Notebook | Description | 
| --- | --- | 
| [als_deep_dive](notebooks/02_modeling/als_deep_dive.ipynb) | Deep dive on the ALS algorithm and implementation.
| [surprise_deep_dive](notebooks/02_modeling/surprise_svd_deep_dive.ipynb) | Deep dive on the SAR algorithm and implementation.

The [Evaluate Notebooks](notebooks/03_evaluate) discuss how to evaluate recommender algorithms for different ranking and rating metrics

| Notebook | Description | 
| --- | --- | 
| [evaluation](notebooks/03_evaluate/evaluation.ipynb) | Examples of different rating and ranking metrics in Python+CPU and PySpark environments.

The [Operationalize Notebooks](notebooks/04_operationalize) discuss how to deploy models in production systems

| Notebook | Description | 
| --- | --- | 
| [als_movie_o16n](notebooks/04_operationalize/als_movie_o16n.ipynb) | End-to-end examples demonstrate how to build, evaluate, and deploye a Spark ALS based movie recommender with Azure services such as [Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction), and [Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/).

## Workflow
The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](/reco_workflow.png)

A few Azure services are recommended for scalable data storage ([Azure Cosmos DB](https://docs.microsoft.com/en-us/azure/cosmos-db/introduction)), model development ([Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/), [Azure Data Science Virtual Machine](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (DSVM), [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service/)), and model opertionalization ([Azure Kubernetes Services](https://azure.microsoft.com/en-us/services/kubernetes-service/) (AKS)). 

![architecture](/reco-arch.png)


## Benchmarks

Here we benchmark the algorithms available in this repository. A notebook for reproducing the benchmarking results can be found [here](notebooks/00_quick_start/benchmark.ipynb).

We benchmark on the Movielens dataset at 100K, 1M, 10M, and 20M sizes. Data is split into train/test sets at at 75/25 ratio and splitting is random. A recommendation model is trained using each of the below collaborative filtering algorithms. We utilize empirical parameter values reported in literature that generated optimal results as reported [here](http://mymedialite.net/examples/datasets.html). We benchmark on a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 K80 GPU). Algorithms that do not apply with GPU accelerations are run on CPU instead. Spark ALS is run in local standalone mode.

**Benchmark results**
<table>
 <tr>
  <th>Dataset</th>
  <th>Algorithm</th>
  <th>Precision</th>
  <th>Recall</th>
  <th>MAP</th>
  <th>NDCG</th>
  <th>RMSE</th>
  <th>MAE</th>
  <th>Explained Variance</th>
  <th>R squared</th>
 </tr>
 <tr>
  <td rowspan=3>Movielens 100k</td>
  <td>ALS</td>
  <td align="right">0.096</td>
  <td align="right">0.079</td>
  <td align="right">0.026</td>
  <td align="right">0.100</td>
  <td align="right">1.110</td>
  <td align="right">0.860</td>
  <td align="right">0.025</td>
  <td align="right">0.023</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">0.958</td>
  <td align="right">0.755</td>
  <td align="right">0.287</td>
  <td align="right">0.287</td>
 </tr>
 <tr>
  <td>SAR Single Node</td>
  <td align="right">0.327</td>
  <td align="right">0.176</td>
  <td align="right">0.106</td>
  <td align="right">0.373</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
 </tr>
 <tr>
  <td rowspan=3>Movielens 1M</td>
  <td>ALS</td>
  <td align="right">0.120</td>
  <td align="right">0.062</td>
  <td align="right">0.022</td>
  <td align="right">0.119</td>
  <td align="right">0.950</td>
  <td align="right">0.735</td>
  <td align="right">0.280</td>
  <td align="right">0.280</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">0.889</td>
  <td align="right">0.697</td>
  <td align="right">0.364</td>
  <td align="right">0.364</td>
 </tr>
 <tr>
  <td>SAR Single Node</td>
  <td align="right">0.277</td>
  <td align="right">0.109</td>
  <td align="right">0.064</td>
  <td align="right">0.308</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
 </tr>
 <tr>
  <td rowspan=3>Movielens 10M</td>
  <td>ALS</td>
  <td align="right">0.090</td>
  <td align="right">0.057</td>
  <td align="right">0.015</td>
  <td align="right">0.084</td>
  <td align="right">0.850</td>
  <td align="right">0.647</td>
  <td align="right">0.359</td>
  <td align="right">0.359</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">0.804</td>
  <td align="right">0.616</td>
  <td align="right">0.424</td>
  <td align="right">0.424</td>
 </tr>
 <tr>
  <td>SAR Single Node</td>
  <td align="right">0.276</td>
  <td align="right">0.156</td>
  <td align="right">0.101</td>
  <td align="right">0.321</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
 </tr>
 <tr>
  <td rowspan=3>Movielens 20M</td>
  <td>ALS</td>
  <td align="right">0.081</td>
  <td align="right">0.052</td>
  <td align="right">0.014</td>
  <td align="right">0.076</td>
  <td align="right">0.830</td>
  <td align="right">0.633</td>
  <td align="right">0.372</td>
  <td align="right">0.371</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">0.790</td>
  <td align="right">0.601</td>
  <td align="right">0.436</td>
  <td align="right">0.436</td>
 </tr>
 <tr >
  <td>SAR Single Node</td>
  <td align="right">0.247</td>
  <td align="right">0.135</td>
  <td align="right">0.085</td>
  <td align="right">0.287</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
  <td align="right">N/A</td>
 </tr>

</table>

**Benchmark comparing time metrics**

In order to benchmark the run-time performance, the algorithms are run on the same data set and the elapsed time for training and testing are collected as follows.

<table>
 <tr>
  <th>Dataset</th>
  <th>Algorithm</th>
  <th>Training time (s)</th>
  <th>Testing time (s)</th>
 </tr>
 <tr>
  <td rowspan=3>Movielens 100k</td>
  <td>ALS</td>
  <td align="right">5.7</td>
  <td align="right">0.3</td>
 </tr>
 <tr >
  <td >Surprise SVD</td>
  <td align="right">13.3</td>
  <td align="right">3.4</td>
 </tr>
 <tr>
  <td>SAR Single Node</td>
  <td align="right">0.7</td>
  <td align="right">0.1</td>
 </tr>
 <tr>
  <td rowspan=3>Movielens 1M</td>
  <td>ALS</td>
  <td align="right">18.0</td>
  <td align="right">0.3</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">129.0</td>
  <td align="right">35.7</td>
 </tr>
 <tr>
  <td>SAR Single Node</td>
  <td align="right">5.8</td>
  <td align="right">0.6</td>
 </tr>
 <tr>
  <td rowspan=3>Movielens 10M</td>
  <td>ALS</td>
  <td align="right">92.0</td>
  <td align="right">0.2</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">1285.0</td>
  <td align="right">253.0</td>
 </tr>
 <tr>
  <td>SAR Single Node</td>
  <td align="right">111.0</td>
  <td align="right">12.6</td>
 </tr>
 <tr>
  <td rowspan=3>Movielens 20M</td>
  <td>ALS</td>
  <td align="right">142.0</td>
  <td align="right">0.3</td>
 </tr>
 <tr>
  <td>Surprise SVD</td>
  <td align="right">2562.0</td>
  <td align="right">506.0</td>
 </tr>
 <tr >
  <td>SAR Single Node</td>
  <td align="right">559.0</td>
  <td align="right">47.3</td>
 </tr>

</table>

## Contributing
This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


## Build Status
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** |  master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792)  || staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | 
| **Linux GPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu?branchName=master)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4997) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_staging?branchName=staging)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4998)|
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4805)|

*NOTE: the tests are executed every night, we use pytest for testing python [utilities]((reco_utils)) and papermill for testing [notebooks](notebooks)*.

