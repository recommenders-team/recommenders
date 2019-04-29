# Benchmarks

In this folder we show benchmarks using different algorithms. To facilitate the benchmark computation, we provide a set of wrapper functions that can be found in the file [benchmark_utils.py](benchmark_utils.py).

The machine we used to perform the benchmarks is a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 P100 GPU). Spark ALS is run in local standalone mode.

## MovieLens

[MovieLens](https://grouplens.org/datasets/movielens/) is one of the most common datasets used in the literature in Recommendation Systems. The dataset consists of a collection of users, movies and movie ratings, there are several available sizes:

* MovieLens 100k: 100,000 ratings from 1000 users on 1700 movies.
* MovieLens 1M: 1 million ratings from 6000 users on 4000 movies.
* MovieLens 10M: 10 million ratings from 72000 users on 10000 movies.
* MovieLens 20M: 20 million ratings from 138000 users on 27000 movies

The MovieLens benchmark can be seen at [movielens.ipynb](movielens.ipynb). In this notebook, the MovieLens dataset is split into training / test sets using a stratified splitting method that takes 75% of each user's ratings as training data, and the remaining 25% ratings as test data. For ranking metrics we use `k=10` (top 10 recommended items). The algorithms used in this benchmark are [ALS](../notebooks/00_quick_start/als_movielens.ipynb), [SVD](../notebooks/02_model/surprise_svd_deep_dive.ipynb), [SAR](../notebooks/00_quick_start/sar_movielens.ipynb), [NCF](../notebooks/00_quick_start/ncf_movielens.ipynb) and [FastAI](../notebooks/00_quick_start/fastai_movielens.ipynb).

|  | Data | Algo | Environment | Train time (s) | Predicting time (s) | RMSE | MAE | R2 | Explained Variance | Recommending time (s) | MAP | nDCG@k | Precision@k | Recall@k |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| 1 | 100k | als | PySpark | 1.4559 | 0.0122 | 0.966788 | 0.754401 | 0.252924 | 0.248931 | 0.0504 | 0.004199 | 0.039901 | 0.043796 | 0.01458 |
| 2 | 100k | svd | Python CPU | 4.8681 | 2.4394 | 0.942123 | 0.744029 | 0.286765 | 0.28679 | 13.0406 | 0.012387 | 0.096353 | 0.091304 | 0.031478 |
| 3 | 100k | sar | Python CPU | 0.2124 | NaN | NaN | NaN | NaN | NaN | 0.1037 |	0.113028 |	0.388321 | 	0.333828 | 0.183179 |
| 4 | 100k | ncf | Python GPU | 54.1899 | NaN | NaN | NaN | NaN | NaN | 2.6587 | 0.10772 | 0.396118 | 0.347296 | 0.180775 |
| 5 | 100k | fastai | Python GPU | 60.5248 | 0.0391 | 0.943084 | 0.744337 | 0.285308 | 0.287671 | 3.1092 | 0.025503 | 0.147866 | 0.130329 | 0.053824 |
| 6 | 1m | als | PySpark | 2.9817 | 0.012 | 0.86033 | 0.679312 | 0.412643 | 0.406617 | 0.0535 | 0.002089 | 0.026097 | 0.032715 | 0.010242 |
| 7 | 1m | svd | Python CPU | 49.5121 | 25.168 | 0.885022 | 0.696763 | 0.372067 | 0.372069 | 199.1151 | 0.008249 | 0.086251 | 0.080487 | 0.02107 |
| 8 | 1m | sar | Python CPU | 2.0946 | NaN | NaN | NaN | NaN | NaN | 2.4326 | 0.066214 | 0.313502 | 0.279692 | 0.111135 |
| 9 | 1m | ncf | Python GPU | 724.9077 | NaN | NaN | NaN | NaN | NaN | 37.7379 | 0.06145 | 0.342406 | 0.314461 | 0.106448 |
| 10 | 1m | fastai | Python GPU | 579.1294 | 0.4028 | 0.874465 | 0.695509 | 0.386959 | 0.389593 | 52.9108 | 0.026113 | 0.184184 | 0.167881 | 0.055633 |
