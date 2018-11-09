
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

<table border=0 cellpadding=0 cellspacing=0 width=1164 style='border-collapse:
 collapse;table-layout:fixed;width:871pt'>
 <col class=xl65 width=111 style='mso-width-source:userset;mso-width-alt:3541;
 width:83pt'>
 <col class=xl65 width=107 style='mso-width-source:userset;mso-width-alt:3413;
 width:80pt'>
 <col class=xl65 width=117 span=2 style='mso-width-source:userset;mso-width-alt:
 3754;width:88pt'>
 <col class=xl68 width=103 style='mso-width-source:userset;mso-width-alt:3285;
 width:77pt'>
 <col class=xl68 width=87 span=7 style='width:65pt'>
 <tr class=xl66 height=21 style='height:16.0pt'>
  <td height=21 class=xl66 width=111 style='height:16.0pt;width:83pt'>Dataset</td>
  <td class=xl66 width=107 style='width:80pt'>Algorithm</td>
  <td class=xl66 width=117 style='width:88pt'>Training time</td>
  <td class=xl66 width=117 style='width:88pt'>Test time</td>
  <td class=xl67 width=103 style='width:77pt'>Precision@10</td>
  <td class=xl67 width=87 style='width:65pt'>Recall@10</td>
  <td class=xl67 width=87 style='width:65pt'>MAP@10</td>
  <td class=xl67 width=87 style='width:65pt'>NDCG@10</td>
  <td class=xl67 width=87 style='width:65pt'>RMSE</td>
  <td class=xl67 width=87 style='width:65pt'>MAE</td>
  <td class=xl67 width=87 style='width:65pt'>Exp Var</td>
  <td class=xl67 width=87 style='width:65pt'>R^2</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=4 height=84 class=xl69 width=111 style='height:64.0pt;width:83pt'>Movielens100k</td>
  <td class=xl65>ALS</td>
  <td class=xl65>5.73 s</td>
  <td class=xl65>326 ms</td>
  <td class=xl68>0.096</td>
  <td class=xl68>0.079</td>
  <td class=xl68>0.026</td>
  <td class=xl68>0.100</td>
  <td class=xl68>1.110</td>
  <td class=xl68>0.860</td>
  <td class=xl68>0.025</td>
  <td class=xl68>0.023</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR PySpark</td>
  <td class=xl65>838 ms</td>
  <td class=xl65>9.56 s</td>
  <td class=xl68>0.327</td>
  <td class=xl68>0.179</td>
  <td class=xl68>0.110</td>
  <td class=xl68>0.379</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR+</td>
  <td class=xl65>7.66 s</td>
  <td class=xl65>16.7 s</td>
  <td class=xl68>0.327</td>
  <td class=xl68>0.176</td>
  <td class=xl68>0.106</td>
  <td class=xl68>0.373</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR CPU</td>
  <td class=xl65>679 ms</td>
  <td class=xl65>116 ms</td>
  <td class=xl68>0.327</td>
  <td class=xl68>0.176</td>
  <td class=xl68>0.106</td>
  <td class=xl68>0.373</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=4 height=84 class=xl65 style='height:64.0pt'>Movielens1M</td>
  <td class=xl65>ALS</td>
  <td class=xl65>18s</td>
  <td class=xl65>339 ms</td>
  <td class=xl68>0.120</td>
  <td class=xl68>0.062</td>
  <td class=xl68>0.022</td>
  <td class=xl68>0.119</td>
  <td class=xl68>0.950</td>
  <td class=xl68>0.735</td>
  <td class=xl68>0.280</td>
  <td class=xl68>0.280</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR PySpark</td>
  <td class=xl65>9.23 s</td>
  <td class=xl65>38.3 s</td>
  <td class=xl68>0.278</td>
  <td class=xl68>0.108</td>
  <td class=xl68>0.064</td>
  <td class=xl68>0.309</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR+</td>
  <td class=xl65></td>
  <td class=xl65></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR CPU</td>
  <td class=xl65>5.83 s</td>
  <td class=xl65>586 ms</td>
  <td class=xl68>0.277</td>
  <td class=xl68>0.109</td>
  <td class=xl68>0.064</td>
  <td class=xl68>0.308</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=4 height=84 class=xl65 style='height:64.0pt'>Movielens10M</td>
  <td class=xl65>ALS</td>
  <td class=xl65>1min 32s</td>
  <td class=xl65>169 ms</td>
  <td class=xl68>0.090</td>
  <td class=xl68>0.057</td>
  <td class=xl68>0.015</td>
  <td class=xl68>0.084</td>
  <td class=xl68>0.850</td>
  <td class=xl68>0.647</td>
  <td class=xl68>0.359</td>
  <td class=xl68>0.359</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR PySpark</td>
  <td class=xl65>1min 26s</td>
  <td class=xl65>5min 10s</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR+</td>
  <td class=xl65></td>
  <td class=xl65></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR CPU</td>
  <td class=xl65>1min 51s</td>
  <td class=xl65>12.6 s</td>
  <td class=xl68>0.276</td>
  <td class=xl68>0.156</td>
  <td class=xl68>0.101</td>
  <td class=xl68>0.321</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td rowspan=4 height=84 class=xl65 style='height:64.0pt'>Movielens20M</td>
  <td class=xl65>ALS</td>
  <td class=xl65>2min 22s</td>
  <td class=xl65>345 ms</td>
  <td class=xl68>0.081</td>
  <td class=xl68>0.052</td>
  <td class=xl68>0.014</td>
  <td class=xl68>0.076</td>
  <td class=xl68>0.830</td>
  <td class=xl68>0.633</td>
  <td class=xl68>0.372</td>
  <td class=xl68>0.371</td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR PySpark</td>
  <td class=xl65></td>
  <td class=xl65></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR+</td>
  <td class=xl65></td>
  <td class=xl65></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <tr height=21 style='height:16.0pt'>
  <td height=21 class=xl65 style='height:16.0pt'>SAR CPU</td>
  <td class=xl65>9min 19s</td>
  <td class=xl65>47.3 s</td>
  <td class=xl68>0.247</td>
  <td class=xl68>0.135</td>
  <td class=xl68>0.085</td>
  <td class=xl68>0.287</td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
  <td class=xl68></td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=111 style='width:83pt'></td>
  <td width=107 style='width:80pt'></td>
  <td width=117 style='width:88pt'></td>
  <td width=117 style='width:88pt'></td>
  <td width=103 style='width:77pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
  <td width=87 style='width:65pt'></td>
 </tr>
 <![endif]>
</table>

## Contributing

This project welcomes contributions and suggestions. Before contributing, please see our [contribution guidelines](CONTRIBUTING.md).


