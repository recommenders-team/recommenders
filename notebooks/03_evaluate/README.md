# Evaluate

In this directory, a notebook is provided to illustrate evaluating models using various performance measures which can be found in [reco_utils](../../reco_utils).

| Notebook | Description | 
| --- | --- | 
| [evaluation](evaluation.ipynb) | Examples of different rating and ranking metrics in Python+CPU and PySpark environments.
| [comparison](comparison.ipynb) | Example of comparing different algorithms for both Rating and Ranking metrics

Two approaches for evaluating model performance are demonstrated along with their respective metrics.
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
    * Arear Under Curver (AUC) - integral area under the receiver operating characteristic curve
    * Logistic loss (Logloss) - the negative log-likelihood of the true labels given the predictions of a classifier

More details on recommender metrics can be found in ths paper by Asela Gunawardana and Guy Shani: [A Survey of Accuracy Evaluation Metrics of Recommendation Tasks
](http://jmlr.csail.mit.edu/papers/volume10/gunawardana09a/gunawardana09a.pdf) or the references as cited in the [evaluation notebook](evaluation.ipynb).
