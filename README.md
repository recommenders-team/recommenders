
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** |  master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792)  || staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | 
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4805)|

*NOTE: the tests are executed every night, we use pytest for testing python [utilities]((reco_utils)) and papermill for testing [notebooks](notebooks)*.

# Recommenders 

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learning to illustrate four key tasks: 
1. Preparing and loading data for each recommender algorithm. 
2. Using different algorithms such as SAR, ALS, etc., for building recommender models. 
3. Evaluating algorithms with offline metrics. 
4. Operationalizing models in a production environment. The examples work across Python + cpu and PySpark environments, and contain guidance as to which algorithm to run in which environment based on scale and other requirements. 

Several utilities are provided in [reco_utils](reco_utils) which will help accelerate experimenting with and building recommendation systems. These utility functions are used to load datasets (i.e., via Pandas DataFrames in python and via Spark DataFrames in Spark) in the manner expected by different algorithms, evaluate different model outputs, split training data, and perform other common tasks. Reference implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications. 

## Environment Setup
* Please see the [setup guide](SETUP.md).

## Notebooks Overview

- The [Quick-Start Notebooks](notebooks/00_quick_start/) detail how you can quickly get up and run with state-of-the-art algorithms such as the Smart Adaptive Recommendation (SAR) algorithm. 
- The [Data Notebooks](notebooks/01_data) detail how to prepare and split data properly for recommendation systems
- The [Modeling Notebooks](notebooks/02_modeling) deep dive into implemetnations of different recommender algorithms
- The [Evaluate Notebooks](notebooks/03_evaluate) discuss how to evaluate recommender algorithms for different ranking and rating metrics
- The [Operationalize Notebooks](notebooks/04_operationalize) discuss how to deploy models in production systems

| Notebook | Description | 
| --- | --- | 
| [als_pyspark_movielens](notebooks/00_quick_start/als_pyspark_movielens.ipynb) | Utilizing the ALS algorithm to power movie ratings in a PySpark environment.
| [sar_python_cpu_movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) | Utilizing the Smart Adaptive Recommendations (SAR) algorithm to power movie ratings in a Python+CPU environment.
| [sar_pyspark_movielens](notebooks/00_quick_start/sar_pyspark_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a PySpark environment.
| [sarplus_movielens](notebooks/00_quick_start/sarplus_movielens.ipynb) | Utilizing the SAR+ algorithm to power movie ratings in a PySpark environment.
| [data_split](notebooks/01_data/data_split.ipynb) | Details on splitting data (randomly, chronologically, etc).
| [als_deep_dive](notebooks/02_modeling/als_deep_dive.ipynb) | Deep dive on the ALS algorithm and implementation.
| [sar_deep_dive](notebooks/02_modeling/sar_deep_dive.ipynb) | Deep dive on the SAR algorithm and implementation.
| [evaluation](notebooks/03_evaluate/evaluation.ipynb) | Examples of different rating and ranking metrics in Python+CPU and PySpark environments.

## Benchmarks

Here we benchmark all the algorithms available in this repository.

<table>
 <tr>
  <td >Dataset</td>
  <td>Algorithm</td>
  <td>Training time</td>
  <td>Test time</td>
  <td>Precision@10</td>
  <td>Recall@10</td>
  <td>MAP@10</td>
  <td>NDCG@10</td>
  <td>RMSE</td>
  <td>MAE</td>
  <td>Exp Var</td>
  <td>R^2</td>
 </tr>
 <tr>
  <td rowspan=4>Movielens100k</td>
  <td>ALS</td>
  <td>5.73 s</td>
  <td>326 ms</td>
  <td>0.096</td>
  <td>0.079</td>
  <td>0.026</td>
  <td>0.100</td>
  <td>1.110</td>
  <td>0.860</td>
  <td>0.025</td>
  <td>0.023</td>
 </tr>
 <tr >
  <td >SAR PySpark</td>
  <td>838 ms</td>
  <td>9.56 s</td>
  <td>0.327</td>
  <td>0.179</td>
  <td>0.110</td>
  <td>0.379</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR+</td>
  <td>7.66 s</td>
  <td>16.7 s</td>
  <td>0.327</td>
  <td>0.176</td>
  <td>0.106</td>
  <td>0.373</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR CPU</td>
  <td>679 ms</td>
  <td>116 ms</td>
  <td>0.327</td>
  <td>0.176</td>
  <td>0.106</td>
  <td>0.373</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td rowspan=4>Movielens1M</td>
  <td>ALS</td>
  <td>18s</td>
  <td>339 ms</td>
  <td>0.120</td>
  <td>0.062</td>
  <td>0.022</td>
  <td>0.119</td>
  <td>0.950</td>
  <td>0.735</td>
  <td>0.280</td>
  <td>0.280</td>
 </tr>
 <tr>
  <td>SAR PySpark</td>
  <td>9.23 s</td>
  <td>38.3 s</td>
  <td>0.278</td>
  <td>0.108</td>
  <td>0.064</td>
  <td>0.309</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR+</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR CPU</td>
  <td>5.83 s</td>
  <td>586 ms</td>
  <td>0.277</td>
  <td>0.109</td>
  <td>0.064</td>
  <td>0.308</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td rowspan=4>Movielens10M</td>
  <td>ALS</td>
  <td>1min 32s</td>
  <td>169 ms</td>
  <td>0.090</td>
  <td>0.057</td>
  <td>0.015</td>
  <td>0.084</td>
  <td>0.850</td>
  <td>0.647</td>
  <td>0.359</td>
  <td>0.359</td>
 </tr>
 <tr>
  <td>SAR PySpark</td>
  <td>1min 26s</td>
  <td>5min 10s</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR+</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR CPU</td>
  <td>1min 51s</td>
  <td>12.6 s</td>
  <td>0.276</td>
  <td>0.156</td>
  <td>0.101</td>
  <td>0.321</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td rowspan=4>Movielens20M</td>
  <td>ALS</td>
  <td>2min 22s</td>
  <td>345 ms</td>
  <td>0.081</td>
  <td>0.052</td>
  <td>0.014</td>
  <td>0.076</td>
  <td>0.830</td>
  <td>0.633</td>
  <td>0.372</td>
  <td>0.371</td>
 </tr>
 <tr>
  <td>SAR PySpark</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR+</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr >
  <td>SAR CPU</td>
  <td>9min 19s</td>
  <td>47.3 s</td>
  <td>0.247</td>
  <td>0.135</td>
  <td>0.085</td>
  <td>0.287</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>

</table>

## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


