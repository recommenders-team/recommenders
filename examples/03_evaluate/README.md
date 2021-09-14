# Evaluate

In this directory, notebooks are provided to illustrate evaluating models using various performance measures which can be found in [recommenders](../../recommenders).

| Notebook | Description | 
| --- | --- | 
| [diversity, novelty etc.](als_movielens_diversity_metrics.ipynb) | Examples of non accuracy based metrics in PySpark environment.
| [evaluation](evaluation.ipynb) | Examples of various rating and ranking metrics in Python+CPU and PySpark environments.

Several approaches for evaluating model performance are demonstrated along with their respective metrics.
1. Rating Metrics: These are used to evaluate how accurate a recommender is at predicting ratings that users gave to items
    * Root Mean Square Error (RMSE) - measure of average error in predicted ratings
    * R Squared (R<sup>2</sup>) - essentially how much of the total variation is explained by the model
    * Mean Absolute Error (MAE) - similar to RMSE but uses absolute value instead of squaring and taking the root of the average
    * Explained Variance - how much of the variance in the data is explained by the model
2. Ranking Metrics: These are used to evaluate how relevant recommendations are for users
    * Precision - this measures the proportion of recommended items that are relevant
    * Recall - this measures the proportion of relevant items that are recommended
    * Normalized Discounted Cumulative Gain (NDCG) - evaluates how well the predicted items for a user are ranked based on relevance
    * Mean Average Precision (MAP) - average precision for each user normalized over all users
3. Classification metrics: These are used to evaluate binary labels
    * Arear Under Curver (AUC) - integral area under the receiver operating characteristic curve
    * Logistic loss (Logloss) - the negative log-likelihood of the true labels given the predictions of a classifier
4. Non accuracy based metrics: These do not compare predictions against ground truth but instead evaluate the following properties of the recommendations
    * Novelty - measures of how novel recommendation items are by calculating their recommendation frequency among users 
    * Diversity - measures of how different items in a set are with respect to each other
    * Serendipity - measures of how surprising recommendations are to to a specific user by comparing them to the items that the user has already interacted with
    * Coverage - measures related to the distribution of items recommended by the system. 
    
References:
1. Asela Gunawardana and Guy Shani: [A Survey of Accuracy Evaluation Metrics of Recommendation Tasks
](http://jmlr.csail.mit.edu/papers/volume10/gunawardana09a/gunawardana09a.pdf)
2. Dimitris Paraschakis et al, "Comparative Evaluation of Top-N Recommenders in e-Commerce: An Industrial Perspective", IEEE ICMLA, 2015, Miami, FL, USA.
3. Yehuda Koren and Robert Bell, "Advances in Collaborative Filtering", Recommender Systems Handbook, Springer, 2015.
4. Chris Bishop, "Pattern Recognition and Machine Learning", Springer, 2006.

