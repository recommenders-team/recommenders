"""
Python evaluation of rating metrics, based on scikit-learn
"""
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    PREDICTION_COL
)


class RatingEvaluation:
    """Python Rating Evaluator"""

    def __init__(
        self,
        rating_true,
        rating_pred,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL,
    ):
        """Initializer.
        Args:
            rating_true (pd.DataFrame): True data. There should be no duplicate (userID, itemID) pairs.
            rating_pred (pd.DataFrame): Predicted data. There should be no duplicate (userID, itemID) pairs.
            col_user (str): column name for user.
            col_item (str): column name for item.
            col_rating (str): column name for rating.
            col_prediction (str): column name for prediction.
        """

        if col_user not in rating_true.columns:
            raise ValueError("Schema of y_true not valid. Missing User Col")
        if col_item not in rating_true.columns:
            raise ValueError("Schema of y_true not valid. Missing Item Col")
        if col_rating not in rating_true.columns:
            raise ValueError("Schema of y_true not valid. Missing Rating Col")

        if col_user not in rating_pred.columns:
            # pragma : No Cover
            raise ValueError("Schema of y_pred not valid. Missing User Col")
        if col_item not in rating_pred.columns:
            # pragma : No Cover
            raise ValueError("Schema of y_pred not valid. Missing Item Col")
        if col_prediction not in rating_pred.columns:
            raise ValueError(
                "Schema of y_true not valid. Missing Prediction Col: "
                + str(rating_pred.columns)
            )

        # Join truth and prediction data frames on userID and itemID
        if col_rating == col_prediction:
            rating_true_pred = pd.merge(
                rating_true,
                rating_pred,
                on=[col_user, col_item],
                suffixes=["_true", "_pred"],
            )
            rating_true_pred.rename(
                columns={col_rating + "_true": DEFAULT_RATING_COL}, inplace=True
            )
            rating_true_pred.rename(
                columns={col_prediction + "_pred": PREDICTION_COL}, inplace=True
            )
        else:
            rating_true_pred = pd.merge(rating_true, rating_pred, on=[col_user, col_item])
            rating_true_pred.rename(columns={col_rating: DEFAULT_RATING_COL}, inplace=True)
            rating_true_pred.rename(columns={col_prediction: PREDICTION_COL}, inplace=True)

        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.rating_true_pred = rating_true_pred

    def rmse(self):
        """Calculate Root Mean Squared Error
        Returns:
            Root mean squared error (float).
        """

        return np.sqrt(
            mean_squared_error(
                self.rating_true_pred[DEFAULT_RATING_COL], self.rating_true_pred[PREDICTION_COL]
            )
        )

    def mae(self):
        """Calculate Mean Absolute Error.
        Returns:
            Mean Absolute Error (float)
        """
        return mean_absolute_error(
            self.rating_true_pred[DEFAULT_RATING_COL], self.rating_true_pred[PREDICTION_COL]
        )

    def rsquared(self):
        """Calculate R squared
        Returns:
            R squared (float <= 1)
        """
        return r2_score(
            self.rating_true_pred[DEFAULT_RATING_COL], self.rating_true_pred[PREDICTION_COL]
        )

    def exp_var(self):
        """Calculate explained variance.
        Returns:
            Explained variance (float <= 1)
        """
        return explained_variance_score(
            self.rating_true_pred[DEFAULT_RATING_COL], self.rating_true_pred[PREDICTION_COL]
        )
