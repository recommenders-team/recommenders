import numpy as np
import pandas as pd

from reco_utils.common.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_PREDICTION_COL


def compute_rating_predictions(algo, data, usercol=DEFAULT_USER_COL, itemcol=DEFAULT_ITEM_COL, predcol=DEFAULT_PREDICTION_COL):
    """
    Computes predictions of an algorithm from Surprise on the data. Can be used for computing rating metrics like RMSE.
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    predictions = [algo.predict(getattr(row, usercol), getattr(row, itemcol)) for row in data.itertuples()]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(index=str, columns={'uid': usercol, 'iid': itemcol, 'est': predcol})
    return predictions.drop(['details', 'r_ui'], axis='columns')


def compute_ranking_predictions(algo, data, usercol=DEFAULT_USER_COL, itemcol=DEFAULT_ITEM_COL,
                            predcol=DEFAULT_PREDICTION_COL, recommend_seen=False):
    """
    Computes predictions of an algorithm from Surprise on all users and items in data. can be used for computing
    ranking metrics like NDCG.
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        recommend_seen (bool): flag to include (user, item) pairs that appear in data
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, predcol
    """
    preds_lst = []
    for user in data[usercol].unique():
        for item in data[itemcol].unique():
            preds_lst.append([user, item, algo.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, predcol])

    if recommend_seen:
        return all_predictions
    else:
        tempdf = pd.concat([data[[usercol, itemcol]],
                            pd.DataFrame(data=np.ones(data.shape[0]), columns=['dummycol'], index=data.index)],
                            axis=1)
        merged = pd.merge(tempdf, all_predictions, on=[usercol, itemcol], how="outer")
        return merged[merged['dummycol'].isnull()].drop('dummycol', axis=1)
