# Quick Start

In this directory, notebooks are provided to perform a quick demonstration of different algorithms such as Alternating Least Squares ([ALS](https://spark.apache.org/docs/latest/api/python/_modules/pyspark/ml/recommendation.html#ALS)) or Simple Algorithm for Recommendation ([SAR](https://github.com/Microsoft/Product-Recommendations/blob/master/doc/sar.md)). The notebooks show how to establish an end-to-end recommendation pipeline that consists of data preparation, model building, and model evaluation by using the utility functions ([reco_utils](../../reco_utils)).

| Notebook | Dataset | Environment | Description |
| --- | --- | --- | --- |
| [als](als_movielens.ipynb) | MovieLens | PySpark | Utilizing ALS algorithm to predict movie ratings in a PySpark environment.
| [dkn](dkn_MIND_dataset.ipynb) | MIND | Python CPU, GPU | Utilizing the Deep Knowledge-Aware Network (DKN) [2] algorithm for news recommendations using information from a knowledge graph, in a Python+GPU (TensorFlow) environment.
| [fastai](fastai_movielens.ipynb) | MovieLens | Python CPU, GPU | Utilizing FastAI recommender to predict movie ratings in a Python+GPU (PyTorch) environment.
| [lightgbm](lightgbm_tinycriteo.ipynb) | Criteo | Python CPU | Utilizing LightGBM Boosting Tree to predict whether or not a user has clicked on an e-commerce ad |
| [lstur](lstur_synthetic.ipynb) | Synthetic Data | Python CPU, GPU | Utilizing the Neural News Recommendation with Long- and Short-term User Representations (LSTUR) [9] for news recommendation, in a Python+GPU (Tensorflow) enviroment.
| [naml](naml_synthetic.ipynb) | Synthetic Data | Python CPU, GPU | Utilizing the Neural News Recommendation with Attentive Multi-View Learning (NAML) [7] to algorithm for news recommendation using news verticle, subverticle, title and body information, in a Python+GPU (Tensorflow) environment.  
| [ncf](ncf_movielens.ipynb) | MovieLens | Python CPU, GPU |  Utilizing Neural Collaborative Filtering (NCF) [1] to predict movie ratings in a Python+GPU (TensorFlow) environment.
| [npa](npa_synthetic.ipynb) | Synthetic Data | Python CPU, GPU | Utilizing the Neural News Recommendation with Personalized Attention (NPA) [10] for news recommendation, in a Python+GPU (Tensorflow) environment.  
| [nrms](nrms_synthetic.ipynb) | Synthetic Data | Python CPU, GPU | Utilizing the Neural News Recommendation with Multi-Head Self-Attention (NRMS) [8] for news recommendation, in a Python+GPU (Tensorflow) environment.  
| [rbm](rbm_movielens.ipynb)| MovieLens | Python CPU, GPU | Utilizing the Restricted Boltzmann Machine (rbm) [4] to predict movie ratings in a Python+GPU (TensorFlow) environment.<br>
| [rlrmc](rlrmc_movielens.ipynb) | Movielens | Python CPU | Utilizing the Riemannian Low-rank Matrix Completion (RLRMC) [6] to predict movie rating in a Python+CPU environment
| [sar](sar_movielens.ipynb) | MovieLens | Python CPU | Utilizing Simple Algorithm for Recommendation (SAR) algorithm to predict movie ratings in a Python+CPU environment.
| [sar_azureml](sar_movielens_with_azureml.ipynb)| MovieLens | Python CPU | An example of how to utilize and evaluate SAR using the [Azure Machine Learning service](https://docs.microsoft.com/azure/machine-learning/service/overview-what-is-azure-ml) (AzureML). It takes the content of the [sar quickstart notebook](sar_movielens.ipynb) and demonstrates how to use the power of the cloud to manage data, switch to powerful GPU machines, and monitor runs while training a model.
| [a2svd](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use A2SVD [11] to predict a set of movies the user is going to interact in a short time. |
| [caser](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use Caser [12] to predict a set of movies the user is going to interact in a short time. |
| [gru4rec](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use GRU4Rec [13] to predict a set of movies the user is going to interact in a short time. |
| [nextitnet](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use NextItNet [14] to predict a set of movies the user is going to interact in a short time. |
| [sli-rec](sequential_recsys_amazondataset.ipynb) | Amazon | Python CPU, GPU | Use SLi-Rec [11] to predict a set of movies the user is going to interact in a short time. |
| [wide-and-deep](wide_deep_movielens.ipynb) | MovieLens | Python CPU, GPU |  Utilizing Wide-and-Deep Model (Wide-and-Deep) [5] to predict movie ratings in a Python+GPU (TensorFlow) environment.
| [xdeepfm](xdeepfm_criteo.ipynb) | Criteo, Synthetic Data | Python CPU, GPU |  Utilizing the eXtreme Deep Factorization Machine (xDeepFM) [3] to learn both low and high order feature interactions for predicting CTR, in a Python+GPU (TensorFlow) environment.

[1] _Neural Collaborative Filtering_, Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu and Tat-Seng Chua. WWW 2017.<br>
[2] _DKN: Deep Knowledge-Aware Network for News Recommendation_, Hongwei Wang, Fuzheng Zhang, Xing Xie and Minyi Guo. WWW 2018.<br>
[3] _xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems_, Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie and Guangzhong Sun. KDD 2018.<br>
[4] _Restricted Boltzmann Machines for Collaborative Filtering_, Ruslan Salakhutdinov, Andriy Mnih and Geoffrey Hinton. ICML 2007.<br>
[5] _Wide & Deep Learning for Recommender Systems_, Heng-Tze Cheng et al., arXiv:1606.07792 2016. <br>
[6] _A unified framework for structured low-rank matrix learning_, Pratik Jawanpuria and Bamdev Mishra, In International Conference on Machine Learning, 2018. <br>
[7] _NAML: Neural News Recommendation with Attentive Multi-View Learning_, Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie. IJCAI 2019.<br>
[8] _NRMS: Neural News Recommendation with Multi-Head Self-Attention_, Chuhan Wu, Fangzhao Wu, Suyu Ge, Tao Qi, Yongfeng Huang, Xing Xie. in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP).<br>
[9] _LSTUR: Neural News Recommendation with Long- and Short-term User Representations_, Mingxiao An, Fangzhao Wu, Chuhan Wu, Kun Zhang, Zheng Liu and Xing Xie. ACL 2019.<br>
[10] _NPA: Neural News Recommendation with Personalized Attention_, Chuhan Wu, Fangzhao Wu, Mingxiao An, Jianqiang Huang, Yongfeng Huang and Xing Xie. KDD 2019, ADS track.<br>
[11] _Adaptive User Modeling with Long and Short-Term Preferences for Personailzed Recommendation_, Zeping Yu, Jianxun Lian, Ahmad Mahmoody, Gongshen Liu and Xing Xie, IJCAI 2019.<br>
[12] _Personalized top-n sequential recommendation via convolutional sequence embedding_, Jiaxi Tang and Ke Wang, ACM WSDM 2018.<br>
[13] _Session-based Recommendations with Recurrent Neural Networks_, Balazs Hidasi, Alexandros Karatzoglou, Linas Baltrunas and Domonkos Tikk, ICLR 2016.<br>
[14] _A Simple Convolutional Generative Network for Next Item Recommendation_, Fajie Yuan, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose and Xiangnan He, WSDM 2019. <br>
