# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

try:
    from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
    from pyspark.sql import Window, DataFrame
    from pyspark.sql.functions import col, row_number, expr
    from pyspark.sql.functions import udf
    import pyspark.sql.functions as F
    from pyspark.sql.types import IntegerType, DoubleType, StructType, StructField
    from pyspark.ml.linalg import VectorUDT
except ImportError:
    pass  # skip this import if we are in pure python environment

from recommenders.utils.constants import (
    DEFAULT_PREDICTION_COL,
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_RELEVANCE_COL,
    DEFAULT_SIMILARITY_COL,
    DEFAULT_ITEM_FEATURES_COL,
    DEFAULT_ITEM_SIM_MEASURE,
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_K,
    DEFAULT_THRESHOLD,
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
        col_prediction=DEFAULT_PREDICTION_COL,
    ):
        """Initializer.

        This is the Spark version of rating metrics evaluator.
        The methods of this class, calculate rating metrics such as root mean squared error, mean absolute error,
        R squared, and explained variance.

        Args:
            rating_true (pyspark.sql.DataFrame): True labels.
            rating_pred (pyspark.sql.DataFrame): Predicted labels.
            col_user (str): column name for user.
            col_item (str): column name for item.
            col_rating (str): column name for rating.
            col_prediction (str): column name for prediction.
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
            col(self.col_user),
            col(self.col_item),
            col(self.col_rating).cast("double").alias("label"),
        )
        self.rating_pred = self.rating_pred.select(
            col(self.col_user),
            col(self.col_item),
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
        """Calculate Root Mean Squared Error.

        Returns:
            float: Root mean squared error.
        """
        return self.metrics.rootMeanSquaredError

    def mae(self):
        """Calculate Mean Absolute Error.

        Returns:
            float: Mean Absolute Error.
        """
        return self.metrics.meanAbsoluteError

    def rsquared(self):
        """Calculate R squared.

        Returns:
            float: R squared.
        """
        return self.metrics.r2

    def exp_var(self):
        """Calculate explained variance.

        .. note::
           Spark MLLib's implementation is buggy (can lead to values > 1), hence we use var().

        Returns:
            float: Explained variance (min=0, max=1).
        """
        var1 = self.y_pred_true.selectExpr("variance(label - prediction)").collect()[0][
            0
        ]
        var2 = self.y_pred_true.selectExpr("variance(label)").collect()[0][0]
        return 1 - var1 / var2


class SparkRankingEvaluation:
    """Spark Ranking Evaluator"""

    def __init__(
        self,
        rating_true,
        rating_pred,
        k=DEFAULT_K,
        relevancy_method="top_k",
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_prediction=DEFAULT_PREDICTION_COL,
        threshold=DEFAULT_THRESHOLD,
    ):
        """Initialization.
        This is the Spark version of ranking metrics evaluator.
        The methods of this class, calculate ranking metrics such as precision@k, recall@k, ndcg@k, and mean average
        precision.

        The implementations of precision@k, ndcg@k, and mean average precision are referenced from Spark MLlib, which
        can be found at `here <https://spark.apache.org/docs/2.3.0/mllib-evaluation-metrics.html#ranking-systems>`_.

        Args:
            rating_true (pyspark.sql.DataFrame): DataFrame of true rating data (in the
                format of customerID-itemID-rating tuple).
            rating_pred (pyspark.sql.DataFrame): DataFrame of predicted rating data (in
                the format of customerID-itemID-rating tuple).
            col_user (str): column name for user.
            col_item (str): column name for item.
            col_rating (str): column name for rating.
            col_prediction (str): column name for prediction.
            k (int): number of items to recommend to each user.
            relevancy_method (str): method for determining relevant items. Possible
                values are "top_k", "by_time_stamp", and "by_threshold".
            threshold (float): threshold for determining the relevant recommended items.
                This is used for the case that predicted ratings follow a known
                distribution. NOTE: this option is only activated if relevancy_method is
                set to "by_threshold".
        """
        self.rating_true = rating_true
        self.rating_pred = rating_pred
        self.col_user = col_user
        self.col_item = col_item
        self.col_rating = col_rating
        self.col_prediction = col_prediction
        self.threshold = threshold

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
            "top_k": _get_top_k_items,
            "by_time_stamp": _get_relevant_items_by_timestamp,
            "by_threshold": _get_relevant_items_by_threshold,
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
                threshold=self.threshold,
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
        """Calculate ranking metrics."""
        self._items_for_user_pred = self.rating_pred

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
        """Get precision@k.

        .. note::
            More details can be found
            `here <http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.precisionAt>`_.

        Return:
            float: precision at k (min=0, max=1)
        """
        precision = self._metrics.precisionAt(self.k)

        return precision

    def recall_at_k(self):
        """Get recall@K.

        .. note::
            More details can be found
            `here <http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.meanAveragePrecision>`_.

        Return:
            float: recall at k (min=0, max=1).
        """
        recall = self._items_for_user_all.rdd.map(
            lambda x: float(len(set(x[0]).intersection(set(x[1])))) / float(len(x[1]))
        ).mean()

        return recall

    def ndcg_at_k(self):
        """Get Normalized Discounted Cumulative Gain (NDCG)

        .. note::
            More details can be found
            `here <http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.ndcgAt>`_.

        Return:
            float: nDCG at k (min=0, max=1).
        """
        ndcg = self._metrics.ndcgAt(self.k)

        return ndcg

    def map_at_k(self):
        """Get mean average precision at k.

        .. note::
            More details can be found
            `here <http://spark.apache.org/docs/2.1.1/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics.meanAveragePrecision>`_.

        Return:
            float: MAP at k (min=0, max=1).
        """
        maprecision = self._metrics.meanAveragePrecision

        return maprecision


def _get_top_k_items(
    dataframe,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=DEFAULT_K,
):
    """Get the input customer-item-rating tuple in the format of Spark
    DataFrame, output a Spark DataFrame in the dense format of top k items
    for each user.

    .. note::
        if it is implicit rating, just append a column of constants to be ratings.

    Args:
        dataframe (pyspark.sql.DataFrame): DataFrame of rating data (in the format of
        customerID-itemID-rating tuple).
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_prediction (str): column name for prediction.
        k (int): number of items for each user.

    Return:
        pyspark.sql.DataFrame: DataFrame of top k items for each user.
    """
    window_spec = Window.partitionBy(col_user).orderBy(col(col_rating).desc())

    # this does not work for rating of the same value.
    items_for_user = (
        dataframe.select(
            col_user, col_item, col_rating, row_number().over(window_spec).alias("rank")
        )
        .where(col("rank") <= k)
        .groupby(col_user)
        .agg(F.collect_list(col_item).alias(col_prediction))
    )

    return items_for_user


def _get_relevant_items_by_threshold(
    dataframe,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    threshold=DEFAULT_THRESHOLD,
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
        col_prediction (str): column name for prediction.
        threshold (float): threshold for determining the relevant recommended items.
            This is used for the case that predicted ratings follow a known
            distribution.

    Return:
        pyspark.sql.DataFrame: DataFrame of customerID-itemID-rating tuples with only relevant
        items.
    """
    items_for_user = (
        dataframe.orderBy(col_rating, ascending=False)
        .where(col_rating + " >= " + str(threshold))
        .select(col_user, col_item, col_rating)
        .withColumn(
            col_prediction, F.collect_list(col_item).over(Window.partitionBy(col_user))
        )
        .select(col_user, col_prediction)
        .dropDuplicates()
    )

    return items_for_user


def _get_relevant_items_by_timestamp(
    dataframe,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_rating=DEFAULT_RATING_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
    col_prediction=DEFAULT_PREDICTION_COL,
    k=DEFAULT_K,
):
    """Get relevant items for each customer defined by timestamp.

    Relevant items are defined as k items that appear mostly recently
    according to timestamps.

    Args:
        dataframe (pyspark.sql.DataFrame): A Spark DataFrame of customerID-itemID-rating-timeStamp
            tuples.
        col_user (str): column name for user.
        col_item (str): column name for item.
        col_rating (str): column name for rating.
        col_timestamp (str): column name for timestamp.
        col_prediction (str): column name for prediction.
        k: number of relevent items to be filtered by the function.

    Return:
        pyspark.sql.DataFrame: DataFrame of customerID-itemID-rating tuples with only relevant items.
    """
    window_spec = Window.partitionBy(col_user).orderBy(col(col_timestamp).desc())

    items_for_user = (
        dataframe.select(
            col_user, col_item, col_rating, row_number().over(window_spec).alias("rank")
        )
        .where(col("rank") <= k)
        .withColumn(
            col_prediction, F.collect_list(col_item).over(Window.partitionBy(col_user))
        )
        .select(col_user, col_prediction)
        .dropDuplicates([col_user, col_prediction])
    )

    return items_for_user


class SparkDiversityEvaluation:
    """Spark Evaluator for diversity, coverage, novelty, serendipity"""

    def __init__(
        self,
        train_df,
        reco_df,
        item_feature_df=None,
        item_sim_measure=DEFAULT_ITEM_SIM_MEASURE,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_relevance=None,
    ):
        """Initializer.

        This is the Spark version of diversity metrics evaluator.
        The methods of this class calculate the following diversity metrics:

        * Coverage - it includes two metrics:
            1. catalog_coverage, which measures the proportion of items that get recommended from the item catalog;
            2. distributional_coverage, which measures how unequally different items are recommended in the
               recommendations to all users.
        * Novelty - A more novel item indicates it is less popular, i.e. it gets recommended less frequently.
        * Diversity - The dissimilarity of items being recommended.
        * Serendipity - The "unusualness" or "surprise" of recommendations to a user. When 'col_relevance' is used,
            it indicates how "pleasant surprise" of recommendations is to a user.

        The metric definitions/formulations are based on the following references with modification:

        :Citation:

            G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
            Recommender Systems Handbook pp. 257-297, 2010.

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist: introducing
            serendipity into music recommendation, WSDM 2012

            P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
            choice, discovery and relevance, ECIR 2011

            Eugene Yan, Serendipity: Accuracy’s unpopular best friend in Recommender Systems,
            eugeneyan.com, April 2020

        Args:
            train_df (pyspark.sql.DataFrame): Data set with historical data for users and items they
                have interacted with; contains col_user, col_item. Assumed to not contain any duplicate rows.
                Interaction here follows the *item choice model* from Castells et al.
            reco_df (pyspark.sql.DataFrame): Recommender's prediction output, containing col_user, col_item,
                col_relevance (optional). Assumed to not contain any duplicate user-item pairs.
            item_feature_df (pyspark.sql.DataFrame): (Optional) It is required only when item_sim_measure='item_feature_vector'.
                It contains two columns: col_item and features (a feature vector).
            item_sim_measure (str): (Optional) This column indicates which item similarity measure to be used.
                Available measures include item_cooccurrence_count (default choice) and item_feature_vector.
            col_user (str): User id column name.
            col_item (str): Item id column name.
            col_relevance (str): Optional. This column indicates whether the recommended item is actually
                relevant to the user or not.
        """

        self.train_df = train_df.select(col_user, col_item)
        self.col_user = col_user
        self.col_item = col_item
        self.sim_col = DEFAULT_SIMILARITY_COL
        self.df_cosine_similarity = None
        self.df_user_item_serendipity = None
        self.df_user_serendipity = None
        self.avg_serendipity = None
        self.df_item_novelty = None
        self.avg_novelty = None
        self.df_intralist_similarity = None
        self.df_user_diversity = None
        self.avg_diversity = None
        self.item_feature_df = item_feature_df
        self.item_sim_measure = item_sim_measure

        if col_relevance is None:
            self.col_relevance = DEFAULT_RELEVANCE_COL
            # relevance term, default is 1 (relevant) for all
            self.reco_df = reco_df.select(
                col_user, col_item, F.lit(1.0).alias(self.col_relevance)
            )
        else:
            self.col_relevance = col_relevance
            self.reco_df = reco_df.select(
                col_user, col_item, F.col(self.col_relevance).cast(DoubleType())
            )

        if self.item_sim_measure == "item_feature_vector":
            self.col_item_features = DEFAULT_ITEM_FEATURES_COL
            required_schema = StructType(
                (
                    StructField(self.col_item, IntegerType()),
                    StructField(self.col_item_features, VectorUDT()),
                )
            )
            if self.item_feature_df is not None:

                if str(required_schema) != str(item_feature_df.schema):
                    raise Exception(
                        "Incorrect schema! item_feature_df should have schema:"
                        + str(required_schema)
                    )
            else:
                raise Exception(
                    "item_feature_df not specified! item_feature_df must be provided "
                    "if choosing to use item_feature_vector to calculate item similarity. "
                    "item_feature_df should have schema:" + str(required_schema)
                )

        # check if reco_df contains any user_item pairs that are already shown in train_df
        count_intersection = (
            self.train_df.select(self.col_user, self.col_item)
            .intersect(self.reco_df.select(self.col_user, self.col_item))
            .count()
        )

        if count_intersection != 0:
            raise Exception(
                "reco_df should not contain any user_item pairs that are already shown in train_df"
            )

    def _get_pairwise_items(self, df):
        """Get pairwise combinations of items per user (ignoring duplicate pairs [1,2] == [2,1])"""
        return (
            df.select(self.col_user, F.col(self.col_item).alias("i1"))
            .join(
                df.select(
                    F.col(self.col_user).alias("_user"),
                    F.col(self.col_item).alias("i2"),
                ),
                (F.col(self.col_user) == F.col("_user")) & (F.col("i1") <= F.col("i2")),
            )
            .select(self.col_user, "i1", "i2")
        )

    def _get_cosine_similarity(self, n_partitions=200):

        if self.item_sim_measure == "item_cooccurrence_count":
            # calculate item-item similarity based on item co-occurrence count
            self._get_cooccurrence_similarity(n_partitions)
        elif self.item_sim_measure == "item_feature_vector":
            # calculate item-item similarity based on item feature vectors
            self._get_item_feature_similarity(n_partitions)
        else:
            raise Exception(
                "item_sim_measure not recognized! The available options include 'item_cooccurrence_count' and 'item_feature_vector'."
            )
        return self.df_cosine_similarity

    def _get_cooccurrence_similarity(self, n_partitions):
        """Cosine similarity metric from

        :Citation:

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
            introducing serendipity into music recommendation, WSDM 2012

        The item indexes in the result are such that i1 <= i2.
        """
        if self.df_cosine_similarity is None:
            pairs = self._get_pairwise_items(df=self.train_df)
            item_count = self.train_df.groupBy(self.col_item).count()

            self.df_cosine_similarity = (
                pairs.groupBy("i1", "i2")
                .count()
                .join(
                    item_count.select(
                        F.col(self.col_item).alias("i1"),
                        F.pow(F.col("count"), 0.5).alias("i1_sqrt_count"),
                    ),
                    on="i1",
                )
                .join(
                    item_count.select(
                        F.col(self.col_item).alias("i2"),
                        F.pow(F.col("count"), 0.5).alias("i2_sqrt_count"),
                    ),
                    on="i2",
                )
                .select(
                    "i1",
                    "i2",
                    (
                        F.col("count")
                        / (F.col("i1_sqrt_count") * F.col("i2_sqrt_count"))
                    ).alias(self.sim_col),
                )
                .repartition(n_partitions, "i1", "i2")
            )
        return self.df_cosine_similarity

    @staticmethod
    @udf(returnType=DoubleType())
    def sim_cos(v1, v2):
        p = 2
        return float(v1.dot(v2)) / float(v1.norm(p) * v2.norm(p))

    def _get_item_feature_similarity(self, n_partitions):
        """Cosine similarity metric based on item feature vectors

        The item indexes in the result are such that i1 <= i2.
        """
        if self.df_cosine_similarity is None:
            self.df_cosine_similarity = (
                self.item_feature_df.select(
                    F.col(self.col_item).alias("i1"),
                    F.col(self.col_item_features).alias("f1"),
                )
                .join(
                    self.item_feature_df.select(
                        F.col(self.col_item).alias("i2"),
                        F.col(self.col_item_features).alias("f2"),
                    ),
                    (F.col("i1") <= F.col("i2")),
                )
                .select("i1", "i2", self.sim_cos("f1", "f2").alias("sim"))
                .sort("i1", "i2")
                .repartition(n_partitions, "i1", "i2")
            )
        return self.df_cosine_similarity

    # Diversity metrics
    def _get_intralist_similarity(self, df):
        """Intra-list similarity from

        :Citation:

            "Improving Recommendation Lists Through Topic Diversification",
            Ziegler, McNee, Konstan and Lausen, 2005.
        """
        if self.df_intralist_similarity is None:
            pairs = self._get_pairwise_items(df=df)
            similarity_df = self._get_cosine_similarity()
            # Fillna(0) is needed in the cases where similarity_df does not have an entry for a pair of items.
            # e.g. i1 and i2 have never occurred together.
            self.df_intralist_similarity = (
                pairs.join(similarity_df, on=["i1", "i2"], how="left")
                .fillna(0)
                .filter(F.col("i1") != F.col("i2"))
                .groupBy(self.col_user)
                .agg(F.mean(self.sim_col).alias("avg_il_sim"))
                .select(self.col_user, "avg_il_sim")
            )
        return self.df_intralist_similarity

    def user_diversity(self):
        """Calculate average diversity of recommendations for each user.
        The metric definition is based on formula (3) in the following reference:

        :Citation:

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
            introducing serendipity into music recommendation, WSDM 2012

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with the following columns: col_user, user_diversity.
        """
        if self.df_user_diversity is None:
            self.df_intralist_similarity = self._get_intralist_similarity(self.reco_df)
            self.df_user_diversity = (
                self.df_intralist_similarity.withColumn(
                    "user_diversity", 1 - F.col("avg_il_sim")
                )
                .select(self.col_user, "user_diversity")
                .orderBy(self.col_user)
            )
        return self.df_user_diversity

    def diversity(self):
        """Calculate average diversity of recommendations across all users.

        Returns:
            float: diversity.
        """
        if self.avg_diversity is None:
            self.df_user_diversity = self.user_diversity()
            self.avg_diversity = self.df_user_diversity.agg(
                {"user_diversity": "mean"}
            ).first()[0]
        return self.avg_diversity

    # Novelty metrics
    def historical_item_novelty(self):
        """Calculate novelty for each item. Novelty is computed as the minus logarithm of
        (number of interactions with item / total number of interactions). The definition of the metric
        is based on the following reference using the choice model (eqs. 1 and 6):

        :Citation:

            P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
            choice, discovery and relevance, ECIR 2011

        The novelty of an item can be defined relative to a set of observed events on the set of all items.
        These can be events of user choice (item "is picked" by a random user) or user discovery
        (item "is known" to a random user). The above definition of novelty reflects a factor of item popularity.
        High novelty values correspond to long-tail items in the density function, that few users have interacted
        with and low novelty values correspond to popular head items.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with the following columns: col_item, item_novelty.
        """
        if self.df_item_novelty is None:
            n_records = self.train_df.count()
            self.df_item_novelty = (
                self.train_df.groupBy(self.col_item)
                .count()
                .withColumn("item_novelty", -F.log2(F.col("count") / n_records))
                .select(self.col_item, "item_novelty")
                .orderBy(self.col_item)
            )
        return self.df_item_novelty

    def novelty(self):
        """Calculate the average novelty in a list of recommended items (this assumes that the recommendation list
        is already computed). Follows section 5 from

        :Citation:

            P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
            choice, discovery and relevance, ECIR 2011

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: novelty.
        """
        if self.avg_novelty is None:
            self.df_item_novelty = self.historical_item_novelty()
            n_recommendations = self.reco_df.count()
            self.avg_novelty = (
                self.reco_df.groupBy(self.col_item)
                .count()
                .join(self.df_item_novelty, self.col_item)
                .selectExpr("sum(count * item_novelty)")
                .first()[0]
                / n_recommendations
            )
        return self.avg_novelty

    # Serendipity metrics
    def user_item_serendipity(self):
        """Calculate serendipity of each item in the recommendations for each user.
        The metric definition is based on the following references:

        :Citation:

            Y.C. Zhang, D.Ó. Séaghdha, D. Quercia and T. Jambor, Auralist:
            introducing serendipity into music recommendation, WSDM 2012

            Eugene Yan, Serendipity: Accuracy’s unpopular best friend in Recommender Systems,
            eugeneyan.com, April 2020

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with columns: col_user, col_item, user_item_serendipity.
        """
        # for every col_user, col_item in reco_df, join all interacted items from train_df.
        # These interacted items are repeated for each item in reco_df for a specific user.
        if self.df_user_item_serendipity is None:
            self.df_cosine_similarity = self._get_cosine_similarity()
            self.df_user_item_serendipity = (
                self.reco_df.select(
                    self.col_user,
                    self.col_item,
                    F.col(self.col_item).alias(
                        "reco_item_tmp"
                    ),  # duplicate col_item to keep
                )
                .join(
                    self.train_df.select(
                        self.col_user, F.col(self.col_item).alias("train_item_tmp")
                    ),
                    on=[self.col_user],
                )
                .select(
                    self.col_user,
                    self.col_item,
                    F.least(F.col("reco_item_tmp"), F.col("train_item_tmp")).alias(
                        "i1"
                    ),
                    F.greatest(F.col("reco_item_tmp"), F.col("train_item_tmp")).alias(
                        "i2"
                    ),
                )
                .join(self.df_cosine_similarity, on=["i1", "i2"], how="left")
                .fillna(0)
                .groupBy(self.col_user, self.col_item)
                .agg(F.mean(self.sim_col).alias("avg_item2interactedHistory_sim"))
                .join(self.reco_df, on=[self.col_user, self.col_item])
                .withColumn(
                    "user_item_serendipity",
                    (1 - F.col("avg_item2interactedHistory_sim"))
                    * F.col(self.col_relevance),
                )
                .select(self.col_user, self.col_item, "user_item_serendipity")
                .orderBy(self.col_user, self.col_item)
            )
        return self.df_user_item_serendipity

    def user_serendipity(self):
        """Calculate average serendipity for each user's recommendations.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: col_user, user_serendipity.
        """
        if self.df_user_serendipity is None:
            self.df_user_item_serendipity = self.user_item_serendipity()
            self.df_user_serendipity = (
                self.df_user_item_serendipity.groupBy(self.col_user)
                .agg(F.mean("user_item_serendipity").alias("user_serendipity"))
                .orderBy(self.col_user)
            )
        return self.df_user_serendipity

    def serendipity(self):
        """Calculate average serendipity for recommendations across all users.

        Returns:
            float: serendipity.
        """
        if self.avg_serendipity is None:
            self.df_user_serendipity = self.user_serendipity()
            self.avg_serendipity = self.df_user_serendipity.agg(
                {"user_serendipity": "mean"}
            ).first()[0]
        return self.avg_serendipity

    # Coverage metrics
    def catalog_coverage(self):
        """Calculate catalog coverage for recommendations across all users.
        The metric definition is based on the "catalog coverage" definition in the following reference:

        :Citation:

            G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
            Recommender Systems Handbook pp. 257-297, 2010.

        Returns:
            float: catalog coverage
        """
        # distinct item count in reco_df
        count_distinct_item_reco = self.reco_df.select(self.col_item).distinct().count()
        # distinct item count in train_df
        count_distinct_item_train = (
            self.train_df.select(self.col_item).distinct().count()
        )

        # catalog coverage
        c_coverage = count_distinct_item_reco / count_distinct_item_train
        return c_coverage

    def distributional_coverage(self):
        """Calculate distributional coverage for recommendations across all users.
        The metric definition is based on formula (21) in the following reference:

        :Citation:

            G. Shani and A. Gunawardana, Evaluating Recommendation Systems,
            Recommender Systems Handbook pp. 257-297, 2010.

        Returns:
            float: distributional coverage
        """
        # In reco_df, how  many times each col_item is being recommended
        df_itemcnt_reco = self.reco_df.groupBy(self.col_item).count()

        # the number of total recommendations
        count_row_reco = self.reco_df.count()
        df_entropy = df_itemcnt_reco.withColumn(
            "p(i)", F.col("count") / count_row_reco
        ).withColumn("entropy(i)", F.col("p(i)") * F.log2(F.col("p(i)")))
        # distributional coverage
        d_coverage = -df_entropy.agg(F.sum("entropy(i)")).collect()[0][0]

        return d_coverage
