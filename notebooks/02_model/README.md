# Model

In this directory, notebooks are provided to give a deep dive into training models using different algorithms such as 
 Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) and Singular Value Decomposition (SVD) using [Surprise](http://surpriselib.com/) python package. The notebooks make use of the utility functions ([reco_utils](../../reco_utils))
 available in the repo.

| Notebook | Description | 
| --- | --- | 
| [als_deep_dive](als_deep_dive.ipynb) | Deep dive on the ALS algorithm and implementation
| [surprise_svd_deep_dive](surprise_svd_deep_dive.ipynb) | Deep dive on a SVD algorithm and implementation
| [sar_single_node_deep_dive](sar_single_node_deep_dive.ipynb) | Deep dive on the SAR algorithm and implementation

Details on model training are best found inside each notebook.