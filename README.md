
| Build Type | Branch | Status |  | Branch | Status | 
| --- | --- | --- | --- | --- | --- | 
| **Linux CPU** |  master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4792)  || staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4594) | 
| **Linux GPU** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu?branchName=master)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4997) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_gpu_staging?branchName=staging)](https://msdata.visualstudio.com/DefaultCollection/AlgorithmsAndDataScience/_build/latest?definitionId=4998)|
| **Linux Spark** | master | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark?branchName=master)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4804) | | staging | [![Status](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_apis/build/status/nightly_spark_staging?branchName=staging)](https://msdata.visualstudio.com/AlgorithmsAndDataScience/_build/latest?definitionId=4805)|

*NOTE: the tests are executed every night, we use pytest for testing python [utilities]((reco_utils)) and papermill for testing [notebooks](notebooks)*.

# Recommenders 

This repository provides examples and best practices for building recommendation systems, provided as Jupyter notebooks. The examples detail our learning to illustrate four key tasks: 
1. Preparing and loading data for each recommender algorithm. 
2. Using different algorithms such as Smart Adapative Recommendation (SAR), Alternating Least Square (ALS), etc., for building recommender models. 
3. Evaluating algorithms with offline metrics. 
4. Operationalizing models in a production environment. The examples work across Python + CPU and PySpark environments, and contain guidance as to which algorithm to run in which environment based on scale and other requirements. 

Several utilities are provided in [reco_utils](reco_utils) which will help accelerate experimenting with and building recommendation systems. These utility functions are used to load datasets (i.e., via Pandas DataFrames in python and via Spark DataFrames in PySpark) in the manner expected by different algorithms, evaluate different model outputs, split training data, and perform other common tasks. Reference implementations of several state-of-the-art algorithms are provided for self-study and customization in your own applications. 

## Environment Setup
* Please see the [setup guide](SETUP.md).

## Notebooks Overview

- The [Quick-Start Notebooks](notebooks/00_quick_start/) detail how you can quickly get up and run with state-of-the-art algorithms such as the SAR algorithm. 
- The [Data Notebooks](notebooks/01_data) detail how to prepare and split data properly for recommendation systems
- The [Modeling Notebooks](notebooks/02_modeling) deep dive into implemetnations of different recommender algorithms
- The [Evaluate Notebooks](notebooks/03_evaluate) discuss how to evaluate recommender algorithms for different ranking and rating metrics
- The [Operationalize Notebooks](notebooks/04_operationalize) discuss how to deploy models in production systems.

| Notebook | Description | 
| --- | --- | 
| [als_pyspark_movielens](notebooks/00_quick_start/als_pyspark_movielens.ipynb) | Utilizing the ALS algorithm to power movie ratings in a PySpark environment.
| [sar_python_cpu_movielens](notebooks/00_quick_start/sar_python_cpu_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a Python + CPU environment.
| [sar_pyspark_movielens](notebooks/00_quick_start/sar_pyspark_movielens.ipynb) | Utilizing the SAR algorithm to power movie ratings in a PySpark environment.
| [sarplus_movielens](notebooks/00_quick_start/sarplus_movielens.ipynb) | Utilizing the SAR+ algorithm to power movie ratings in a PySpark environment.
| [data_split](notebooks/01_data/data_split.ipynb) | Details on splitting data (randomly, chronologically, etc).
| [als_deep_dive](notebooks/02_modeling/als_deep_dive.ipynb) | Deep dive on the ALS algorithm and implementation.
| [sar_deep_dive](notebooks/02_modeling/sar_deep_dive.ipynb) | Deep dive on the SAR algorithm and implementation.
| [surprise_svd_deep_dive](notebooks/02_modeling/surprise_svd_deep_dive.ipynb) | Deep dive on Surprise SVD algorithm and implementation.
| [evaluation](notebooks/03_evaluate/evaluation.ipynb) | Examples of different rating and ranking metrics in Python + CPU and PySpark environments.
| [ALS_databricks_o16n](notebooks/04_operationalize/ALS_Movie_Example.ipynb) | Operationalization of ALS algorithm on Databricks using Azure ML and Kubernetes.

## Benchmarks

Here we benchmark all the algorithms available in this repository.

**NOTES**:
* Time for training and testing is measured in second.
* Ranking metrics (i.e., precision, recall, MAP, and NDCG) are evaluated with k equal to 10. They are not applied to SAR-family algorithms (SAR PySpark, SAR+, and SAR CPU) because these algorithms do not predict explicit ratings that have the same scale with those in the original input data.
* The machine we used is an [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) Standard NC6s_v2 with 6 vcpus, 112 GB memory and 1 K80 GPU.

<table>
 <tr>
  <th>Dataset</th>
  <th>Algorithm</th>
  <th>Training time</th>
  <th>Testing time</th>
  <th>Precision</th>
  <th>Recall</th>
  <th>MAP</th>
  <th>NDCG</th>
  <th>RMSE</th>
  <th>MAE</th>
  <th>Exp Var</th>
  <th>R squared</th>
 </tr>
 <tr>
  <td rowspan=4>Movielens 100k</td>
  <td>ALS</td>
  <td>5.730</td>
  <td>0.326</td>
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
  <td>0.838</td>
  <td>9.560</td>
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
  <td>7.660</td>
  <td>16.700</td>
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
  <td>0.679</td>
  <td>0.116</td>
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
  <td rowspan=4>Movielens 1M</td>
  <td>ALS</td>
  <td>18.000</td>
  <td>0.339</td>
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
  <td>9.230</td>
  <td>38.300</td>
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
  <td>38.000</td>
  <td>108.000</td>
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
  <td>SAR CPU</td>
  <td>5.830</td>
  <td>0.586</td>
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
  <td rowspan=4>Movielens 10M</td>
  <td>ALS</td>
  <td>92.000</td>
  <td>0.169</td>
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
  <td>170.000</td>
  <td>80.000</td>
  <td>0.256</td>
  <td>0.129</td>
  <td>0.081</td>
  <td>0.295</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr>
  <td>SAR CPU</td>
  <td>111.000</td>
  <td>12.600</td>
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
  <td rowspan=4>Movielens 20M</td>
  <td>ALS</td>
  <td>142.000</td>
  <td>0.345</td>
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
  <td>400.000</td>
  <td>221.000</td>
  <td>0.203</td>
  <td>0.071</td>
  <td>0.041</td>
  <td>0.226</td>
  <td></td>
  <td></td>
  <td></td>
  <td></td>
 </tr>
 <tr >
  <td>SAR CPU</td>
  <td>559.000</td>
  <td>47.300</td>
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


