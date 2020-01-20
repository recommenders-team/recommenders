# Model

In this directory, notebooks are provided to give a deep dive into training models using different algorithms such as
 Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) and Singular Value Decomposition (SVD) using [Surprise](http://surpriselib.com/) python package. The notebooks make use of the utility functions ([reco_utils](../../reco_utils))
 available in the repo.

| Notebook | Environment | Description |
| --- | --- | --- |
| [als_deep_dive](als_deep_dive.ipynb) | PySpark | Deep dive on the ALS algorithm and implementation.
| [mmlspark_lightgbm_criteo](mmlspark_lightgbm_criteo.ipynb) | PySpark | LightGBM gradient boosting tree algorithm implementation in MML Spark with Criteo dataset.
| [baseline_deep_dive](baseline_deep_dive.ipynb) | --- | Deep dive on baseline performance estimation.
| [ncf_deep_dive](ncf_deep_dive.ipynb) | Python CPU, GPU | Deep dive on a NCF algorithm and implementation.
| [rbm_deep_dive](rbm_deep_dive.ipynb)| Python CPU, GPU | Deep dive on the rbm algorithm and its implementation.
| [cornac_bpr_deep_dive](cornac_bpr_deep_dive.ipynb) | Python CPU | Deep dive on the BPR algorithm and implementation.
| [sar_deep_dive](sar_deep_dive.ipynb) | Python CPU | Deep dive on the SAR algorithm and implementation.
| [surprise_svd_deep_dive](surprise_svd_deep_dive.ipynb) | Python CPU | Deep dive on a SVD algorithm and implementation.
| [vowpal_wabbit_deep_dive](vowpal_wabbit_deep_dive.ipynb) | Python CPU | Deep dive into using Vowpal Wabbit for regression and matrix factorization.
| [fm_deep_dive](fm_deep_dive.ipynb) | Python CPU | Deep dive into factorization machine (FM) and field-aware FM (FFM) algorithm.

Details on model training are best found inside each notebook.
