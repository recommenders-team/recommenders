# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd


def compute_test_results(model, train, test, rating_metrics, ranking_metrics):
    """Compute the test results using a trained NCF model.
    
    Args:
        model (object): TF model.
        train (pandas.DataFrame): Train set.
        test (pandas.DataFrame): Test set.
        rating_metrics (list): List of rating metrics.
        ranking_metrics (list): List of ranking metrics.
        
    Returns:
        dict: Test results.
    
    """
    test_results = {}
    
    # Rating Metrics
    predictions = [[row.userID, row.itemID, model.predict(row.userID, row.itemID)]
           for (_, row) in test.iterrows()]

    predictions = pd.DataFrame(predictions, columns=['userID', 'itemID', 'prediction'])
    predictions = predictions.astype({'userID': 'int64', 'itemID': 'int64', 'prediction': 'float64'})

    for metric in rating_metrics:
        test_results[metric] = eval(metric)(test, predictions)
        
    # Ranking Metrics
    users, items, preds = [], [], []
    item = list(train.itemID.unique())
    for user in train.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
        preds.extend(list(model.predict(user, item, is_list=True)))

    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items, "prediction": preds})

    merged = pd.merge(train, all_predictions, on=["userID", "itemID"], how="outer")
    all_predictions = merged[merged.rating.isnull()].drop('rating', axis=1)

    for metric in ranking_metrics:
        test_results[metric] = eval(metric)(test, all_predictions, col_prediction='prediction', k=K)
        
    return test_results
  
  
def combine_metrics_dicts(*metrics):
    """Combine metrics from dicts.
    
    Args:
        metrics (dict): Metrics
        
    Returns:
        pandas.DataFrame: Dataframe with metrics combined.
    """
    df = pd.DataFrame(metrics[0], index=[0])
    for metric in metrics[1:]:
        df = df.append(pd.DataFrame(metric, index=[0]), sort=False)
    return df
