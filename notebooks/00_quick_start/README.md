# Quick Start

In this directory, notebooks are provided to demonstrate the use of different algorithms such as 
 Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) and Smart Adaptive Recommendations ([SAR](https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md)). The notebooks show how to establish an end-to-end recommendation pipeline that consists of
data preparation, model building, and model evaluation by using the utility functions ([reco_utils](../../reco_utils))
 available in the repo.
 
 | Notebook | Description | 
| --- | --- | 
| [als_pyspark_movielens](als_pyspark_movielens.ipynb) | Utilizing the ALS algorithm to predict movie ratings in a PySpark environment.
| [sar_python_cpu_movielens](sar_single_node_movielens.ipynb) | Utilizing the Smart Adaptive Recommendations (SAR) algorithm to power movie ratings in a Python+CPU environment.

