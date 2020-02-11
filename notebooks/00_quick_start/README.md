# Quick Start

In this directory, notebooks are provided to perform a quick demonstration of different algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or Simple Algorithm for Recommendation ([SAR](https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md)). The notebooks show how to establish an end-to-end recommendation pipeline that consists of data preparation, model building, and model evaluation by using the utility functions ([reco_utils](../../reco_utils))
 available in the repo.

| Notebook | Dataset | Environment | Description |
|----------|---------|-------------|-------------|
| [als](als_movielens.ipynb) | MovieLens | PySpark | Use ALS algorithm to predict movie ratings in a PySpark environment. |
| [dkn](dkn_synthetic.ipynb) | Synthetic Data | Python CPU, GPU | Use the Deep Knowledge-Aware Network (DKN) [2] algorithm for news recommendations using information from a knowledge graph, in a Python+GPU (TensorFlow) environment. |
| [fastai](fastai_movielens.ipynb) | MovieLens | Python CPU, GPU | Use FastAI recommender to predict movie ratings in a Python+GPU (PyTorch) environment. |
| [lightgbm](lightgbm_tinycriteo.ipynb) | Criteo | Python CPU | Use LightGBM Boosting Tree to predict whether or not a user has clicked on an e-commerce ad |
| [ncf](ncf_movielens.ipynb) | MovieLens | Python CPU, GPU |  Use Neural Collaborative Filtering (NCF) [1] to predict movie ratings in a Python+GPU (TensorFlow) environment. |
| [rbm](rbm_movielens.ipynb)| MovieLens | Python CPU, GPU | Use the Restricted Boltzmann Machine (rbm) [4] to predict movie ratings in a Python+GPU (TensorFlow) environment. |
| [rlrmc](rlrmc_movielens.ipynb) | Movielens | Python CPU | Use the Riemannian Low-rank Matrix Completion (RLRMC) [6] to predict movie rating in a Python+CPU environment |
| [sar](sar_movielens.ipynb) | MovieLens | Python CPU | Use Simple Algorithm for Recommendation (SAR) algorithm to predict movie ratings in a Python+CPU environment. |
| [sar_azureml](sar_movielens_with_azureml.ipynb) | MovieLens | Python CPU | An example of how to utilize and evaluate SAR using the [Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/overview-what-is-azure-ml) (AzureML). It takes the content of the [sar quickstart notebook](sar_movielens.ipynb) and demonstrates how to use the power of the cloud to manage data, switch to powerful GPU machines, and monitor runs while training a model. |
| [a2svd](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use A2SVD [7] to predict a set of movies the user is going to interact in a short time. |
| [caser](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use Caser [8] to predict a set of movies the user is going to interact in a short time. |
| [gru4rec](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use GRU4Rec [9] to predict a set of movies the user is going to interact in a short time. |
| [sli-rec](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use SLi-Rec [7] to predict a set of movies the user is going to interact in a short time. |
| [wide-and-deep](wide_deep_movielens.ipynb) | MovieLens | Python CPU, GPU |  Use Wide-and-Deep Model (Wide-and-Deep) [5] to predict movie ratings in a Python+GPU (TensorFlow) environment. |
| [xdeepfm](xdeepfm_criteo.ipynb) | Criteo, Synthetic Data | Python CPU, GPU |  Use the eXtreme Deep Factorization Machine (xDeepFM) [3] to learn both low and high order feature interactions for predicting CTR, in a Python+GPU (TensorFlow) environment. |

[1] _Neural Collaborative Filtering_, Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua, WWW 2017.<br>
[2] _DKN: Deep Knowledge-Aware Network for News Recommendation_, Hongwei Wang, Fuzheng Zhang, Xing Xie and Minyi Guo, WWW 2018.<br>
[3] _xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems_, Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie and Guangzhong Sun, KDD 2018.<br>
[4] _Restricted Boltzmann Machines for Collaborative Filtering_, Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton, ICML 2007.<br>
[5] _Wide & Deep Learning for Recommender Systems_, Heng-Tze Cheng et al., arXiv:1606.07792 2016.<br>
[6] _A unified framework for structured low-rank matrix learning_, Pratik Jawanpuria and Bamdev Mishra, ICML 2018.<br>
[7] _Adaptive User Modeling with Long and Short-Term Preferences for Personailzed Recommendation_, Z. Yu, J. Lian, A. Mahmoody, G. Liu and X. Xie, IJCAI 2019.<br>
[8] _Personalized top-n sequential recommendation via convolutional sequence embedding_, J. Tang and K. Wang, ACM WSDM 2018.<br>
[9] _Session-based Recommendations with Recurrent Neural Networks_, B. Hidasi, A. Karatzoglou, L. Baltrunas and D. Tikk, ICLR 2016.<br>

