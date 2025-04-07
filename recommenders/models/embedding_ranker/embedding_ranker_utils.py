# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import numpy as np
import pandas as pd
import torch

from recommenders.utils.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_PREDICTION_COL,
    DEFAULT_K,
)

def predict_rating(
    model,
    test_df,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    batch_size=1024,
):
    """Predict ratings for user-item pairs in test data.
    
    Args:
        model (NNEmbeddingRanker): Trained embedding ranker model.
        test_df (pandas.DataFrame): Test dataframe containing user-item pairs.
        col_user (str): User column name.
        col_item (str): Item column name.
        col_rating (str): Rating column name.
        col_prediction (str): Prediction column name.
        batch_size (int): Batch size for predictions to avoid memory issues.
        
    Returns:
        pandas.DataFrame: Dataframe with user, item, prediction columns.
    """
    # Create a copy of the test data with only the needed columns
    test_copy = test_df[[col_user, col_item]].copy()
    
    # Process in batches to avoid memory issues
    predictions = []
    for i in range(0, len(test_copy), batch_size):
        batch = test_copy.iloc[i:i+batch_size]
        users = batch[col_user].values
        items = batch[col_item].values
        batch_predictions = model.predict(users, items, is_list=True)
        predictions.extend(batch_predictions)
    
    # Add predictions to the dataframe
    test_copy[col_prediction] = predictions
    
    return test_copy

def generate_recommendations(
    model,
    train_df,
    test_df=None,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    top_k=DEFAULT_K,
    remove_seen=True,
    batch_size=1024,
):
    """Generate top-k recommendations for all users or users in test data.
    
    Args:
        model (NNEmbeddingRanker): Trained embedding ranker model.
        train_df (pandas.DataFrame): Training dataframe used to identify seen items if remove_seen=True.
        test_df (pandas.DataFrame, optional): Test dataframe to identify users to generate recommendations for.
                                             If None, recommendations are generated for all users in training.
        col_user (str): User column name.
        col_item (str): Item column name.
        col_prediction (str): Prediction column name.
        top_k (int): Number of top items to recommend.
        remove_seen (bool): Whether to remove items that appear in the training data.
        batch_size (int): Batch size for predictions to avoid memory issues.
        
    Returns:
        pandas.DataFrame: Dataframe with user, item, prediction columns for top-k items per user.
    """
    # If test_df is provided, use users from test set
    # Otherwise, use all users from training
    if test_df is not None:
        users = test_df[col_user].unique()
    else:
        users = train_df[col_user].unique()
    
    # Get all items
    all_items = list(model.item_id_map.keys())
    
    # Filter users that are not in the training set
    valid_users = [user for user in users if user in model.user_id_map]
    
    if not valid_users:
        raise ValueError("No valid users found in the dataset")
    
    # Create combinations of users and items to predict
    all_pairs = []
    for user in valid_users:
        for item in all_items:
            all_pairs.append((user, item))
    
    # Process in batches to avoid memory issues
    all_predictions = []
    for i in range(0, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i+batch_size]
        batch_users = [pair[0] for pair in batch_pairs]
        batch_items = [pair[1] for pair in batch_pairs]
        batch_predictions = model.predict(batch_users, batch_items, is_list=True)
        
        # Create batch results
        for j, (user, item) in enumerate(batch_pairs):
            all_predictions.append({
                col_user: user,
                col_item: item,
                col_prediction: batch_predictions[j]
            })
    
    # Convert to dataframe
    result_df = pd.DataFrame(all_predictions)
    
    # If remove_seen is True, remove items that appear in the training data
    if remove_seen:
        # Create a set of user-item pairs from the training data
        seen_pairs = set(zip(train_df[col_user], train_df[col_item]))
        
        # Filter out seen pairs
        result_df = result_df[~result_df.apply(lambda row: (row[col_user], row[col_item]) in seen_pairs, axis=1)]
    
    # Get top-k recommendations for each user
    top_k_df = (
        result_df
        .sort_values([col_user, col_prediction], ascending=[True, False])
        .groupby(col_user).head(top_k)
        .reset_index(drop=True)
    )
    
    return top_k_df

def evaluate_model(
    model,
    test_df,
    metrics,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=DEFAULT_K,
    batch_size=1024,
):
    """Evaluate model using rating and ranking metrics.
    
    Args:
        model (NNEmbeddingRanker): Trained embedding ranker model.
        test_df (pandas.DataFrame): Test dataframe.
        metrics (dict): Dictionary of metric functions to use for evaluation.
            Keys should be metric names and values should be functions that take
            test_df, predictions_df, and other parameters.
        col_user (str): User column name.
        col_item (str): Item column name.
        col_rating (str): Rating column name.
        col_prediction (str): Prediction column name.
        k (int): K value for ranking metrics.
        batch_size (int): Batch size for predictions to avoid memory issues.
        
    Returns:
        dict: Dictionary with metric name as key and metric value as value.
    """
    # Generate predictions for test data
    predictions_df = predict_rating(
        model,
        test_df,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        batch_size=batch_size,
    )
    
    # Calculate metrics
    results = {}
    for metric_name, metric_func in metrics.items():
        # Different metrics may have different required parameters
        if 'k' in metric_func.__code__.co_varnames:
            results[metric_name] = metric_func(
                test_df,
                predictions_df,
                col_user=col_user,
                col_item=col_item,
                col_rating=col_rating,
                col_prediction=col_prediction,
                k=k
            )
        else:
            results[metric_name] = metric_func(
                test_df,
                predictions_df,
                col_user=col_user,
                col_item=col_item,
                col_rating=col_rating,
                col_prediction=col_prediction
            )
    
    return results 