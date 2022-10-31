# Benchmarks

In this folder we show benchmarks using different algorithms. To facilitate the benchmark computation, we provide a set of wrapper functions that can be found in the file [benchmark_utils.py](benchmark_utils.py).

## MovieLens

[MovieLens](https://grouplens.org/datasets/movielens/) is one of the most common datasets used in the literature in Recommendation Systems. The dataset consists of a collection of users, movies and movie ratings, there are several available sizes:

* MovieLens 100k: 100,000 ratings from 1000 users on 1700 movies.
* MovieLens 1M: 1 million ratings from 6000 users on 4000 movies.
* MovieLens 10M: 10 million ratings from 72000 users on 10000 movies.
* MovieLens 20M: 20 million ratings from 138000 users on 27000 movies

The MovieLens benchmark can be seen at [movielens.ipynb](movielens.ipynb). This illustrative comparison applies to collaborative filtering algorithms available in this repository such as [Spark ALS](../00_quick_start/als_movielens.ipynb), [SVD](../02_model_collaborative_filtering/surprise_svd_deep_dive.ipynb), [SAR](../00_quick_start/sar_movielens.ipynb), [LightGCN](../02_model_collaborative_filtering/lightgcn_deep_dive.ipynb) and others using the Movielens dataset, using three environments (CPU, GPU and Spark). These algorithms are usable in a variety of recommendation tasks, including product or news recommendations.
