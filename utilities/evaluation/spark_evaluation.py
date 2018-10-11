"""
SparkEvaluation
"""

from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.sql import Window, DataFrame
from pyspark.sql.functions import col, row_number, expr

from utilities.common.constants import (
    PREDICTION_COL,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
)


class SparkRatingEvaluation:
    """Spark Rating Evaluator"""
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
            rating_true (spark.DataFrame): True labels.
            rating_pred (spark.DataFrame): Predicted labels.
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        # Check if inputs are Spark DataFrames.
        if not isinstance(self.rating_true, DataFrame):
            raise TypeError(
                "rating_true should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        if not isinstance(self.rating_pred, DataFrame):
            raise TypeError(
                "rating_pred should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        # Check if columns exist.
        true_columns = self.rating_true.columns
        pred_columns = self.rating_pred.columns

        if rating_true.count() == 0:
            raise ValueError("Empty input dataframe")
        if rating_pred.count() == 0:
            raise ValueError("Empty input dataframe")

        if self.col_user not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing User Col")
        if self.col_item not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Item Col")
        if self.col_rating not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Rating Col")

        if self.col_user not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing User Col"
            )  # pragma : No Cover
        if self.col_item not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing Item Col"
            )  # pragma : No Cover
        if self.col_prediction not in pred_columns:
            raise ValueError("Schema of rating_pred not valid. Missing Prediction Col")

        self.rating_true = self.rating_true.select(
            col(self.col_user).cast("double"),
            col(self.col_item).cast("double"),
            col(self.col_rating).cast("double").alias("label"),
        )
        self.rating_pred = self.rating_pred.select(
            col(self.col_user).cast("double"),
            col(self.col_item).cast("double"),
            col(self.col_prediction).cast("double").alias("prediction"),
        )

        self.y_pred_true = (
            self.rating_true.join(
                self.rating_pred, [self.col_user, self.col_item], "inner"
            )
            .drop(self.col_user)
            .drop(self.col_item)
        )

        self.metrics = RegressionMetrics(
            self.y_pred_true.rdd.map(lambda x: (x.prediction, x.label))
        )

    def rmse(self):
        """Calculate Root Mean Squared Error
        Returns:
            Root Mean Squared Error
        """
        return self.metrics.rootMeanSquaredError

    def rsquared(self):
        """Calculate R squared
        Returns:
            R squared
        """
        return self.metrics.r2

    def mae(self):
        """Calculate Mean Absolute for data
        Returns:
            Mean Absolute Error
        """
        return self.metrics.meanAbsoluteError

    def exp_var(self):
        """Calculate explained variance.
        Returns:
            Explained variance
            = 1 - var(y-hat(y))/var(y)
        NOTE: Spark MLLib's implementation is buggy (can lead to values > 1), hence we use var()
        """
        var1 = self.y_pred_true.selectExpr("variance(label - prediction)").collect()[0][
            0
        ]
        var2 = self.y_pred_true.selectExpr("variance(label)").collect()[0][0]
        return 1 - var1 / var2

    @staticmethod
    def get_available_metrics():
        """Get all available metrics.

        Return:
            List of metric names.
        """
        return ["rmse", "rsquare", "mae", "exp_var"]

    def get_metric(self, metrics="rmse", _all=False):
        """Get metrics.

        Args:
            metrics (string or string list): metrics to obtain.
            _all (bool): whether to return all metrics.

        Return:
            dictionary object that contains the specified metrics.
        """
        metrics_to_calculate = self.get_available_metrics() if _all else metrics

        metrics_dict = {
            "rmse": self.rmse,
            "mae": self.mae,
            "rsquare": self.rsquared,
            "exp_var": self.exp_var,
        }

        metrics_output = {}
        if isinstance(metrics_to_calculate, list):
            for metric in metrics_to_calculate:
                metrics_output[metric] = metrics_dict[metric]()
        elif isinstance(metrics_to_calculate, str):
            metrics_output[metrics_to_calculate] = metrics_dict[metrics_to_calculate]()
        else:
            raise TypeError(
                "The input metric argument type is not valid. Only a string or a list of strings "
                "are allowed."
            )

        return metrics_output


class SparkRankingEvaluation:
    """SparkRankingEvaluation"""

    def __init__(
        self,
        rating_true,
        rating_pred,
        k=10,
        relevancy_method="top_k",
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=PREDICTION_COL,
    ):
        """Initialization.

        Args:
            rating_true (Spark DataFrame): DataFrame of true rating data (in the
            format of customerID-itemID-rating tuple).
            rating_pred (Spark DataFrame): DataFrame of predicted rating data (in
            the format of customerID-itemID-rating tuple).
            col_user (str): column name for user.
            col_item (str): column name for item.
            col_rating (str): column name for rating.
            col_prediction (str): column name for prediction.
            k (int): number of items to recommend to each user.
            relevancy_method (str): method for determining relevant items.
            Possible values are "top_k", "by_time_stamp", and "by_threshold".
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction

        # Check if inputs are Spark DataFrames.
        if not isinstance(self.rating_true, DataFrame):
            raise TypeError(
                "rating_true should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        if not isinstance(self.rating_pred, DataFrame):
            raise TypeError(
                "rating_pred should be but is not a Spark DataFrame"
            )  # pragma : No Cover

        # Check if columns exist.
        true_columns = self.rating_true.columns
        pred_columns = self.rating_pred.columns

        if self.col_user not in true_columns:
            raise ValueError(
                "Schema of rating_true not valid. Missing User Col: "
                + str(true_columns)
            )
        if self.col_item not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Item Col")
        if self.col_rating not in true_columns:
            raise ValueError("Schema of rating_true not valid. Missing Rating Col")

        if self.col_user not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing User Col"
            )  # pragma : No Cover
        if self.col_item not in pred_columns:
            raise ValueError(
                "Schema of rating_pred not valid. Missing Item Col"
            )  # pragma : No Cover
        if self.col_prediction not in pred_columns:
            raise ValueError("Schema of rating_pred not valid. Missing Prediction Col")

        self.k = k

        relevant_func = {
            "top_k": get_top_k_items,
            "by_time_stamp": get_relevant_items_by_timestamp,
            "by_threshold": get_relevant_items_by_threshold,
        }

        if relevancy_method not in relevant_func:
            raise ValueError(
                "relevancy_method should be one of {}".format(
                    list(relevant_func.keys())
                )
            )

        self.rating_pred = (
            relevant_func[relevancy_method](
                dataframe=self.rating_pred,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_prediction,
            )
            if relevancy_method == "by_threshold"
            else relevant_func[relevancy_method](
                dataframe=self.rating_pred,
                col_user=self.col_user,
                col_item=self.col_item,
                col_rating=self.col_prediction,
                k=self.k,
            )
        )

        self._metrics = self._calculate_metrics()

    def _calculate_metrics(self):
        """Calculate ranking metrics.
        """
        self._items_for_user_pred = (
            self.rating_pred.groupBy(self.col_user)
            .agg(expr("collect_list(" + self.col_item + ") as prediction"))
            .select(self.col_user, "prediction")
        )

        self._items_for_user_true = (
            self.rating_true.groupBy(self.col_user)
            .agg(expr("collect_list(" + self.col_item + ") as ground_truth"))
            .select(self.col_user, "ground_truth")
        )

        self._items_for_user_all = self._items_for_user_pred.join(
            self._items_for_user_true, on=self.col_user
        ).drop(self.col_user)

        return RankingMetrics(self._items_for_user_all.rdd)

    def precision_at_k(self):
        # pylint: disable=line-too-long
        """Get precision@k.

        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation
        .RankingMetrics.precisionAt

        Return:
            precision at k.
        """
        precision = self._metrics.precisionAt(self.k)

        return precision

    def ndcg_at_k(self):
        # pylint: disable=line-too-long
        """Get Normalized Discounted Cumulative Gain (NDCG)@k.

        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation
        .RankingMetrics.ndcgAt

        Return:
            NDCG at k.
        """
        ndcg = self._metrics.ndcgAt(self.k)

        return ndcg

    def map_at_k(self):
        # pylint: disable=line-too-long
        """
        Get mean average precision at k.

        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation
        .RankingMetrics.meanAveragePrecision

        Return:
            MAP at k.
        """
        maprecision = self._metrics.meanAveragePrecision

        return maprecision

    def recall_at_k(self):
        # pylint: disable=line-too-long
        """
        Get mean average precision at k.

        More details can be found at
        http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation
        .RankingMetrics.meanAveragePrecision

        Return:
            Recall at k.
        """
        recall = self._items_for_user_all.rdd.map(
            lambda x: float(len(set(x[0]).intersection(set(x[1])))) / float(len(x[1]))
        ).mean()

        return recall

    @staticmethod
    def get_available_metrics():
        """Get all available metrics.

        Return:
            List of metric names.
        """
        return ["precision@k", "recall@k", "map@k", "ndcg@k"]

    def get_metric(self, metrics="precision@k", _all=False):
        """Get metrics.

        Args:
            metrics (string or string list): metrics to obtain.
            _all (bool): whether to return all metrics.

        Return:
            dictionary object that contains the specified metrics.
        """
        metrics_to_calculate = self.get_available_metrics() if _all else metrics

        metrics_dict = {
            "precision@k": self.precision_at_k,
            "recall@k": self.recall_at_k,
            "map@k": self.map_at_k,
            "ndcg@k": self.ndcg_at_k,
        }

        metrics_output = {}
        if isinstance(metrics_to_calculate, list):
            for metric in metrics_to_calculate:
                metrics_output[metric] = metrics_dict[metric]()
        elif isinstance(metrics_to_calculate, str):
            metrics_output[metrics_to_calculate] = metrics_dict[metrics_to_calculate]()
        else:
            raise TypeError(
                "The input metric argument type is not valid. Only a string or a list of strings "
                "are allowed."
            )

        return metrics_output


def get_top_k_items(
    dataframe, col_user="customerID", col_item="itemID", col_rating="rating", k=10
):
    """Get the input customer-item-rating tuple in the format of Spark
    DataFrame, output a Spark DataFrame in the dense format of top k items
    for each user.
    Note:
        if it is implicit rating, just append a column of constants to be
        ratings.

    Args:
        dataframe (spark.DataFrame): DataFrame of rating data (in the format of
        customerID-itemID-rating tuple).
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        k (int): number of items for each user.

    Return:
        Spark DataFrame of top k items for each user.
    """
    window_spec = Window.partitionBy(col_user).orderBy(col(col_rating).desc())

    # this does not work for rating of the same value.
    items_for_user = (
        dataframe.select(
            col_user, col_item, col_rating, row_number().over(window_spec).alias("rank")
        )
        .where(col("rank") <= k)
        .drop("rank")
    )

    return items_for_user


def get_relevant_items_by_threshold(
    dataframe,
    col_user="customerID",
    col_item="itemID",
    col_rating="rating",
    threshold=3.5,
):
    """Get relevant items for each customer in the input rating data.

    Relevant items are defined as those having ratings above certain threshold.
    The threshold is defined as a statistical measure of the ratings for a
    user, e.g., median.

    Args:
        dataframe: Spark DataFrame of customerID-itemID-rating tuples.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        threshold: threshold for determining the relevant recommended items.
        This is used for the case that predicted ratings follow a known
        distribution.

    Return:
        Spark DataFrame of customerID-itemID-rating tuples with only relevant
        items.
    """
    items_for_user = dataframe.where(col_rating + " >= " + str(threshold)).select(
        col_user, col_item, col_rating
    )

    return items_for_user


def get_relevant_items_by_timestamp(
    dataframe,
    col_user="customerID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timeStamp",
    k=10,
):
    """Get relevant items for each customer defined by timestamp.

    Relevant items are defined as k items that appear mostly recently
    according to timestamps.

    Args:
        dataframe: A Spark DataFrame of customerID-itemID-rating-timeStamp
        tuples.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_timestamp (str): column name for timestamp.
        k: number of relevent items to be filtered by the function.

    Return:
        Spark DataFrame of customerID-itemID-rating tuples with only relevant items.
    """
    window_spec = Window.partitionBy(col_user).orderBy(col(col_timestamp).desc())

    items_for_user = (
        dataframe.select(
            col_user, col_item, col_rating, row_number().over(window_spec).alias("rank")
        )
        .where(col("rank") <= k)
        .drop("rank")
    )

    return items_for_user
