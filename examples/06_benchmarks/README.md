# Benchmarks

In this folder we show benchmarks using different algorithms. To facilitate the benchmark computation, we provide a set of wrapper functions that can be found in the file [benchmark_utils.py](benchmark_utils.py).

The machine we used to perform the benchmarks is a Standard NC6s_v2 [Azure DSVM](https://azure.microsoft.com/en-us/services/virtual-machines/data-science-virtual-machines/) (6 vCPUs, 112 GB memory and 1 P100 GPU). Spark ALS is run in local standalone mode.

## MovieLens

[MovieLens](https://grouplens.org/datasets/movielens/) is one of the most common datasets used in the literature in Recommendation Systems. The dataset consists of a collection of users, movies and movie ratings, there are several available sizes:

* MovieLens 100k: 100,000 ratings from 1000 users on 1700 movies.
* MovieLens 1M: 1 million ratings from 6000 users on 4000 movies.
* MovieLens 10M: 10 million ratings from 72000 users on 10000 movies.
* MovieLens 20M: 20 million ratings from 138000 users on 27000 movies

The MovieLens benchmark can be seen at [movielens.ipynb](movielens.ipynb). In this notebook, the MovieLens dataset is split into training / test sets using a stratified splitting method that takes 75% of each user's ratings as training data, and the remaining 25% ratings as test data. For ranking metrics we use `k=10` (top 10 recommended items). The algorithms used in this benchmark are [ALS](../00_quick_start/als_movielens.ipynb), [SVD](../02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb), [SAR](../00_quick_start/sar_movielens.ipynb), [NCF](../00_quick_start/ncf_movielens.ipynb), [BPR](../02_model_collaborative_filtering/cornac_bpr_deep_dive.ipynb), [BiVAE](../02_model_collaborative_filtering/cornac_bivae_deep_dive.ipynb), [LightGCN](../02_model_collaborative_filtering/lightgcn_deep_dive.ipynb) and [FastAI](../00_quick_start/fastai_movielens.ipynb).


