# Quick Start

In this directory, notebooks are provided to demonstrate the use of different algorithms such as
 Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) and Smart Adaptive Recommendations ([SAR](https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md)). The notebooks show how to establish an end-to-end recommendation pipeline that consists of
data preparation, model building, and model evaluation by using the utility functions ([reco_utils](../../reco_utils))
 available in the repo.

| Notebook | Description |
| --- | --- |
| [als_pyspark_movielens](als_pyspark_movielens.ipynb) | Utilizing ALS algorithm to predict movie ratings in a PySpark environment.
| [fastai_recommendation](fastai_recommendation.ipynb) | Utilizing FastAI recommender to predict movie ratings in a Python+GPU (PyTorch) environment.
| [ncf_movielens](ncf_movielens.ipynb) | Utilizing Neural Collaborative Filtering (NCF) [1] to predict movie ratings in a Python+GPU (TensorFlow) environment.
| [sar_python_cpu_movielens](sar_single_node_movielens.ipynb) | Utilizing Smart Adaptive Recommendations (SAR) algorithm to predict movie ratings in a Python+CPU environment.
| [dkn](dkn.ipynb) | Utilizing the Deep Knowledge-Aware Network (DKN) [2] algorithm for news recommendations using information from a knowledge graph, in a Python+GPU (TensorFlow) environment.
| [xdeepfm](xdeepfm.ipynb) | Utilizing the eXtreme Deep Factorization Machine (xDeepFM) [3] to learn both low and high order feature interactions for predicting CTR, in a Python+GPU (TensorFlow) environment.
| [rbm](rbm_movielens.py)| Utiliziing the Restricted Boltzmann Machine (rbm) [4] to predict movie ratings in Python+GPU (TensorFlow) enviroment.<br>

[1] _Neural Collaborative Filtering_, Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua. WWW 2017.<br>
[2] _DKN: Deep Knowledge-Aware Network for News Recommendation_, Hongwei Wang, Fuzheng Zhang, Xing Xie and Minyi Guo. WWW 2018.<br>
[3] _xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems_, Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie and Guangzhong Sun. KDD 2018.<br>
[4] _Restricted Boltzmann Machines for Collaborative Filtering_, Ruslan Salakhutdinov Andriy Mnih Geoffrey Hinton. ICML 2007.
