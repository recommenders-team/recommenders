import pandas as pd


def compute_predictions(algo, data, usercol='userID', itemcol='itemID'):
    """
    Computes predictions of an algorithm from Surprise on the data
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data on which to predict
        usercol (str): name of the user column
        itemcol (str): name of the item column
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, prediction
    """
    predictions = [algo.predict(row.userID, row.itemID) for (_, row) in data.iterrows()]
    predictions = pd.DataFrame(predictions)
    predictions = predictions.rename(index=str, columns={'uid': usercol, 'iid': itemcol, 'est': 'prediction'})
    return predictions.drop(['details', 'r_ui'], axis='columns')


def compute_all_predictions(algo, data, usercol='userID', itemcol='itemID', ratingcol='rating', recommend_seen=False):
    """
    Computes predictions of an algorithm from Surprise on all users and items in data.
    Args:
        algo (surprise.prediction_algorithms.algo_base.AlgoBase): an algorithm from Surprise
        data (pd.DataFrame): the data from which to get the users and items
        usercol (str): name of the user column
        itemcol (str): name of the item column
        ratingcol (str): name of the rating column
        recommend_seen (bool): flag to include (user, item) pairs that appear in data
    Returns:
        pd.DataFrame: dataframe with usercol, itemcol, prediction
    """
    preds_lst = []
    for user in data[usercol].unique():
        for item in data[itemcol].unique():
            preds_lst.append([user, item, algo.predict(user, item).est])

    all_predictions = pd.DataFrame(data=preds_lst, columns=[usercol, itemcol, "prediction"])

    merged = pd.merge(data, all_predictions, on=[usercol, itemcol], how="outer")
    if recommend_seen:
        return merged.drop(ratingcol, axis=1)
    else:
        return merged[merged.rating.isnull()].drop(ratingcol, axis=1)
