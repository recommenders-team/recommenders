
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** |  master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792)  || staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | 
| **Linux GPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu?branchName=master)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4997) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_staging?branchName=staging)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4998)|
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4805)|

*NOTE: the tests are executed every night, we use pytest for testing python [utilities]((reco_utils)) and papermill for testing [notebooks](notebooks)*.

# Recommenders 

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learning to illustrate four key tasks: 
1. Preparing and loading data for each recommender algorithm. 
2. Using different algorithms such as Smart Adaptive Recommendation (SAR), Alternating Least Square (ALS), etc., for building recommender models. 
3. Evaluating algorithms with offline metrics. 
4. Operationalizing models in a production environment. The examples work across Python + CPU and PySpark environments, and contain guidance as to which algorithm to run in which environment based on scale and other requirements.

Several utilities are provided in [reco_utils](reco_utils) which will help accelerate experimenting with and building recommendation systems. These utility functions are used to load datasets (i.e., via Pandas DataFrames in python and via Spark DataFrames in PySpark) in the manner expected by different algorithms, evaluate different model outputs, split training data, and perform other common tasks. Reference implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications.

The diagram below depicts how the best-practice examples help researchers / developers in the recommendation system development workflow.

![workflow](https://zhledata.blob.core.windows.net/misc/recommender_workflow.png)

## Environment Setup
* Please see the [setup guide](SETUP.md).

## Notebooks Overview

- The [Quick-Start Notebooks](notebooks/00_quick_start/) detail how you can quickly get up and run some recommendation algorithms such as the SAR algorithm. 
- The [Data Notebooks](notebooks/01_data) detail how to prepare and split data properly for recommendation systems.
- The [Modeling Notebooks](notebooks/02_modeling) deep dive into implementations of different recommender algorithms.
- The [Evaluate Notebooks](notebooks/03_evaluate) discuss how to evaluate recommender algorithms for different ranking and rating metrics.
- The [Operationalize Notebooks](notebooks/04_operationalize) discuss how to deploy models in production.

| Notebook | Description | 
| --- | --- | 
| [als_pyspark_movielens](notebooks/00_quick_start/als_pyspark_movielens.ipynb) | Utilizing the ALS algorithm to power movie ratings in a PySpark environment.
| [sar_single_node_movielens](notebooks/00_quick_start/sar_single_node_movielens.ipynb) | Utilizing the SAR Single Node algorithm to power movie ratings in a Python + CPU environment.
| [data_split](notebooks/01_data/data_split.ipynb) | Details on splitting data (randomly, chronologically, etc).
| [als_deep_dive](notebooks/02_modeling/als_deep_dive.ipynb) | Deep dive on the ALS algorithm and implementation.
| [sar_deep_dive](notebooks/02_modeling/sar_deep_dive.ipynb) | Deep dive on the SAR algorithm and implementation.
| [surprise_svd_deep_dive](notebooks/02_modeling/surprise_svd_deep_dive.ipynb) | Deep dive on Surprise SVD algorithm and implementation.
| [evaluation](notebooks/03_evaluate/evaluation.ipynb) | Examples of different rating and ranking metrics in Python + CPU and PySpark environments.
| [ALS_databricks_o16n](notebooks/04_operationalize/ALS_Movie_Example.ipynb) | Operationalization of ALS algorithm on Databricks using Azure ML and Kubernetes.

The recommended environments to run the Spark algorithms is [Azure Databricks](https://azure.microsoft.com/en-us/services/databricks/). The recommended one for using the CPU algorithms is an [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/).

## Benchmarks

Here we benchmark the algorithms available in this repository.

Following list settings used for the benchmarking experimentation:
* The machine we used for the benchmark is an [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/).
    * The size of the DSVM is Standard NC6s_v2. It has 6 vCPUs, 112 GB memory and 1 K80 GPU.
    * Algorithms that do not apply with GPU accelerations are run on CPU instead. Spark ALS is run in local standalone mode.
* Ranking metrics (i.e., precision, recall, MAP, and NDCG) are evaluated with k equal to 10.
* SAR Single Node only has ranking metrics because these algorithms do not predict explicit ratings with the same scale of those in the original input data. Surprise SVD only has rating metrics.
* The hyper parameters of the algorithms are:
   * `ALS(rank=40,maxIter=15,alpha=0.1,regParam=0.01,coldStartStrategy='drop',nonnegative=True)`
   * `SVD(random_state=0, n_factors=200, n_epochs=30, verbose=True)`
   * `SARSingleNodeReference(remove_seen=True, similarity_type="jaccard", time_decay_coefficient=30, time_now=None, timedecay_formula=True)`
   * **NOTE**: We selected these parameters to roughly indicate the performance of these algorithms. However, the parameters we used are not necessarily optimal.

**Benchmark comparing performance metrics**
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
  <td align="right">0.407</td>
  <td align="right">0.323</td>
  <td align="right">0.871</td>
  <td align="right">0.871</td>
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
  <td align="right">0.487</td>
  <td align="right">0.383</td>
  <td align="right">0.810</td>
  <td align="right">0.810</td>
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
  <td align="right">0.557</td>
  <td align="right">0.430</td>
  <td align="right">0.724</td>
  <td align="right">0.724</td>
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
  <td align="right">0.574</td>
  <td align="right">0.440</td>
  <td align="right">0.702</td>
  <td align="right">0.702</td>
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


