# Quick Start

In this directory, notebooks are provided to demonstrate the use of different algorithms such as
 Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) and Smart Adaptive Recommendations ([SAR](https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md)). The notebooks show how to establish an end-to-end recommendation pipeline that consists of
data preparation, model building, and model evaluation by using the utility functions ([reco_utils](../../reco_utils))
 available in the repo.

| Notebook | Dataset | Environment | Description |
| --- | --- | --- | --- |
| [als](als_movielens.ipynb) | MovieLens | PySpark | Utilizing ALS algorithm to predict movie ratings in a PySpark environment.
| [dkn](dkn_synthetic.ipynb) | Synthetic Data | Python CPU, GPU | Utilizing the Deep Knowledge-Aware Network (DKN) [2] algorithm for news recommendations using information from a knowledge graph, in a Python+GPU (TensorFlow) environment.
| [fastai](fastai_movielens.ipynb) | MovieLens | Python CPU, GPU | Utilizing FastAI recommender to predict movie ratings in a Python+GPU (PyTorch) environment.
| [lightgbm](lightgbm_tinycriteo.ipynb) | Criteo | Python CPU | Utilizing LightGBM Boosting Tree to predict whether or not a user has clicked on an e-commerce ad |
| [ncf](ncf_movielens.ipynb) | MovieLens | Python CPU, GPU |  Utilizing Neural Collaborative Filtering (NCF) [1] to predict movie ratings in a Python+GPU (TensorFlow) environment.
| [rbm](rbm_movielens.ipynb)| MovieLens | Python CPU, GPU | Utilizing the Restricted Boltzmann Machine (rbm) [4] to predict movie ratings in a Python+GPU (TensorFlow) environment.<br>
| [sar](sar_movielens.ipynb) | MovieLens | Python CPU | Utilizing Smart Adaptive Recommendations (SAR) algorithm to predict movie ratings in a Python+CPU environment.
| [sar_azureml](sar_movielens_with_azureml.ipynb)| MovieLens | Python CPU | An example of how to utilize and evaluate SAR using the [Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/overview-what-is-azure-ml)(AzureML). It takes the content of the [sar quickstart notebook](sar_movielens.ipynb) and demonstrates how to use the power of the cloud to manage data, switch to powerful GPU machines, and monitor runs while training a model.
| [wide-and-deep](wide_deep_movielens.ipynb) | MovieLens | Python CPU, GPU |  Utilizing Wide-and-Deep Model (Wide-and-Deep) [5] to predict movie ratings in a Python+GPU (TensorFlow) environment.
| [xdeepfm](xdeepfm_criteo.ipynb) | Criteo, Synthetic Data | Python CPU, GPU |  Utilizing the eXtreme Deep Factorization Machine (xDeepFM) [3] to learn both low and high order feature interactions for predicting CTR, in a Python+GPU (TensorFlow) environment.

[1] _Neural Collaborative Filtering_, Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua. WWW 2017.<br>
[2] _DKN: Deep Knowledge-Aware Network for News Recommendation_, Hongwei Wang, Fuzheng Zhang, Xing Xie and Minyi Guo. WWW 2018.<br>
[3] _xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems_, Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie and Guangzhong Sun. KDD 2018.<br>
[4] _Restricted Boltzmann Machines for Collaborative Filtering_, Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton. ICML 2007.<br>
[5] _Wide & Deep Learning for Recommender Systems_, Heng-Tze Cheng et al., arXiv:1606.07792 2016.
