# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import numpy as np
import seaborn as sns

from lightfm.evaluation import precision_at_k, recall_at_k


def model_perf_plots(df):
    """Function to plot model performance metrics.

    Args:
        df (pandas.DataFrame): Dataframe in tidy format, with ['epoch','level','value'] columns

    Returns:
        object: matplotlib axes
    """
    g = sns.FacetGrid(df, col="metric", hue="stage", col_wrap=2, sharey=False)
    g = g.map(sns.scatterplot, "epoch", "value").add_legend()


def compare_metric(df_list, metric="prec", stage="test"):
    """Function to combine and prepare list of dataframes into tidy format.

    Args:
        df_list (list): List of dataframes
        metrics (str): name of metric to be extracted, optional
        stage (str): name of model fitting stage to be extracted, optional

    Returns:
        pandas.DataFrame: Metrics
    """
    colnames = ["model" + str(x) for x in list(range(1, len(df_list) + 1))]
    models = [
        df[(df["stage"] == stage) & (df["metric"] == metric)]["value"]
        .reset_index(drop=True)
        .values
        for df in df_list
    ]

    output = pd.DataFrame(zip(*models), columns=colnames).stack().reset_index()
    output.columns = ["epoch", "data", "value"]
    return output


def track_model_metrics(
    model,
    train_interactions,
    test_interactions,
    k=10,
    no_epochs=100,
    no_threads=8,
    show_plot=True,
    **kwargs
):
    """Function to record model's performance at each epoch, formats the performance into tidy format,
    plots the performance and outputs the performance data.

    Args:
        model (LightFM instance): fitted LightFM model
        train_interactions (scipy sparse COO matrix): train interactions set
        test_interactions (scipy sparse COO matrix): test interaction set
        k (int): number of recommendations, optional
        no_epochs (int): Number of epochs to run, optional
        no_threads (int): Number of parallel threads to use, optional
        **kwargs: other keyword arguments to be passed down

    Returns:
        pandas.DataFrame, LightFM model, matplotlib axes:
        - Performance traces of the fitted model
        - Fitted model
        - Side effect of the method
    """
    # initialising temp data storage
    model_prec_train = [0] * no_epochs
    model_prec_test = [0] * no_epochs

    model_rec_train = [0] * no_epochs
    model_rec_test = [0] * no_epochs

    # fit model and store train/test metrics at each epoch
    for epoch in range(no_epochs):
        model.fit_partial(
            interactions=train_interactions, epochs=1, num_threads=no_threads, **kwargs
        )
        model_prec_train[epoch] = precision_at_k(
            model, train_interactions, k=k, **kwargs
        ).mean()
        model_prec_test[epoch] = precision_at_k(
            model, test_interactions, k=k, **kwargs
        ).mean()

        model_rec_train[epoch] = recall_at_k(
            model, train_interactions, k=k, **kwargs
        ).mean()
        model_rec_test[epoch] = recall_at_k(
            model, test_interactions, k=k, **kwargs
        ).mean()

    # collect the performance metrics into a dataframe
    fitting_metrics = pd.DataFrame(
        zip(model_prec_train, model_prec_test, model_rec_train, model_rec_test),
        columns=[
            "model_prec_train",
            "model_prec_test",
            "model_rec_train",
            "model_rec_test",
        ],
    )
    # convert into tidy format
    fitting_metrics = fitting_metrics.stack().reset_index()
    fitting_metrics.columns = ["epoch", "level", "value"]
    # exact the labels for each observation
    fitting_metrics["stage"] = fitting_metrics.level.str.split("_").str[-1]
    fitting_metrics["metric"] = fitting_metrics.level.str.split("_").str[1]
    fitting_metrics.drop(["level"], axis=1, inplace=True)
    # replace the metric keys to improve visualisation
    metric_keys = {"prec": "Precision", "rec": "Recall"}
    fitting_metrics.metric.replace(metric_keys, inplace=True)
    # plots the performance data
    if show_plot:
        model_perf_plots(fitting_metrics)
    return fitting_metrics, model


def similar_users(user_id, user_features, model, N=10):
    """Function to return top N similar users based on https://github.com/lyst/lightfm/issues/244#issuecomment-355305681

     Args:
        user_id (int): id of user to be used as reference
        user_features (scipy sparse CSR matrix): user feature matric
        model (LightFM instance): fitted LightFM model
        N (int): Number of top similar users to return

    Returns:
        pandas.DataFrame: top N most similar users with score
    """
    _, user_representations = model.get_user_representations(features=user_features)

    # Cosine similarity
    scores = user_representations.dot(user_representations[user_id, :])
    user_norms = np.linalg.norm(user_representations, axis=1)
    user_norms[user_norms == 0] = 1e-10
    scores /= user_norms

    best = np.argpartition(scores, -(N + 1))[-(N + 1) :]
    return pd.DataFrame(
        sorted(zip(best, scores[best] / user_norms[user_id]), key=lambda x: -x[1])[1:],
        columns=["userID", "score"],
    )


def similar_items(item_id, item_features, model, N=10):
    """Function to return top N similar items
    based on https://github.com/lyst/lightfm/issues/244#issuecomment-355305681

    Args:
        item_id (int): id of item to be used as reference
        item_features (scipy sparse CSR matrix): item feature matric
        model (LightFM instance): fitted LightFM model
        N (int): Number of top similar items to return

    Returns:
        pandas.DataFrame: top N most similar items with score
    """
    _, item_representations = model.get_item_representations(features=item_features)

    # Cosine similarity
    scores = item_representations.dot(item_representations[item_id, :])
    item_norms = np.linalg.norm(item_representations, axis=1)
    item_norms[item_norms == 0] = 1e-10
    scores /= item_norms

    best = np.argpartition(scores, -(N + 1))[-(N + 1) :]
    return pd.DataFrame(
        sorted(zip(best, scores[best] / item_norms[item_id]), key=lambda x: -x[1])[1:],
        columns=["itemID", "score"],
    )


def prepare_test_df(test_idx, uids, iids, uid_map, iid_map, weights):
    """Function to prepare test df for evaluation

    Args:
        test_idx (slice): slice of test indices
        uids (numpy.ndarray): Array of internal user indices
        iids (numpy.ndarray): Array of internal item indices
        uid_map (dict): Keys to map internal user indices to external ids.
        iid_map (dict): Keys to map internal item indices to external ids.
        weights (numpy.float32 coo_matrix): user-item interaction

    Returns:
        pandas.DataFrame: user-item selected for testing
    """
    test_df = pd.DataFrame(
        zip(
            uids[test_idx],
            iids[test_idx],
            [list(uid_map.keys())[x] for x in uids[test_idx]],
            [list(iid_map.keys())[x] for x in iids[test_idx]],
        ),
        columns=["uid", "iid", "userID", "itemID"],
    )

    dok_weights = weights.todok()
    test_df["rating"] = test_df.apply(lambda x: dok_weights[x.uid, x.iid], axis=1)

    return test_df[["userID", "itemID", "rating"]]


def prepare_all_predictions(
    data,
    uid_map,
    iid_map,
    interactions,
    model,
    num_threads,
    user_features=None,
    item_features=None,
):
    """Function to prepare all predictions for evaluation.
    Args:
        data (pandas df): dataframe of all users, items and ratings as loaded
        uid_map (dict): Keys to map internal user indices to external ids.
        iid_map (dict): Keys to map internal item indices to external ids.
        interactions (np.float32 coo_matrix): user-item interaction
        model (LightFM instance): fitted LightFM model
        num_threads (int): number of parallel computation threads
        user_features (np.float32 csr_matrix): User weights over features
        item_features (np.float32 csr_matrix):  Item weights over features
    Returns:
        pandas.DataFrame: all predictions
    """
    users, items, preds = [], [], []  # noqa: F841
    item = list(data.itemID.unique())
    for user in data.userID.unique():
        user = [user] * len(item)
        users.extend(user)
        items.extend(item)
    all_predictions = pd.DataFrame(data={"userID": users, "itemID": items})
    all_predictions["uid"] = all_predictions.userID.map(uid_map)
    all_predictions["iid"] = all_predictions.itemID.map(iid_map)

    dok_weights = interactions.todok()
    all_predictions["rating"] = all_predictions.apply(
        lambda x: dok_weights[x.uid, x.iid], axis=1
    )

    all_predictions = all_predictions[all_predictions.rating < 1].reset_index(drop=True)
    all_predictions = all_predictions.drop("rating", axis=1)

    all_predictions["prediction"] = all_predictions.apply(
        lambda x: model.predict(
            user_ids=np.array([x["uid"]], dtype=np.int32),
            item_ids=np.array([x["iid"]], dtype=np.int32),
            user_features=user_features,
            item_features=item_features,
            num_threads=num_threads,
        )[0],
        axis=1,
    )

    return all_predictions[["userID", "itemID", "prediction"]]
