# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pyspark.sql.types import *
from pyspark.sql import functions as F


class DiversityEvaluator:
    def __init__(
        self,
        train_df,
        reco_df,
        user_col="UserId",
        item_col="ItemId",
        relevance_col=None,
    ):
        """Diversity evaluator.
        train (train_df) and recommendation (reco_df) dataframes should already be groupped by user-item pair.

        Metrics includes:
            Coverage - The proportion of items that can be recommended. It includes two metrics: catalog_coverage and distributional_coverage.
            Novelty - A more novel item indicates it is less popular.
            Diversity - The dissimilarity of items being recommended.
            Serendipity - The “unusualness” or “surprise” of recommendations to a user.


        Args:
            train_df (pySpark DataFrame): Training set used for the recommender,
                containing user_col, item_col.
            reco_df (pySpark DataFrame): Recommender's prediction output,
                containing user_col, item_col, relevance_col (optional).
            user_col (str): User id column name.
            item_col (str): Item id column name.
            relevance_col (str): this column indicates whether the recommended item is actually relevent to the user or not.
        """

        self.train_df = train_df.select(user_col, item_col)
        self.user_col = user_col
        self.item_col = item_col
        self.sim_col = "sim"
        self.df_cosine_similariy = None
        self.df_user_item_serendipity = None
        self.df_user_serendipity = None
        self.df_serendipity = None
        self.df_item_novelty = None
        self.df_user_novelty = None
        self.df_novelty = None
        self.df_intralist_similarity = None
        self.df_user_diversity = None
        self.df_diversity = None

        if relevance_col is None:
            self.relevance_col = "relevance"
            # relevance term, default is 1 (relevent) for all
            self.reco_df = reco_df.select(
                user_col, item_col, F.lit(1.0).alias(self.relevance_col)
            )
        else:
            self.relevance_col = relevance_col
            self.reco_df = reco_df.select(
                user_col, item_col, F.col(self.relevance_col).cast(DoubleType())
            )

        # check if reco_df contain any user_item pairs that are already shown train_df
        count_intersection = (
            self.train_df.select(self.user_col, self.item_col)
            .intersect(self.reco_df.select(self.user_col, self.item_col))
            .count()
        )

        if count_intersection != 0:
            raise Exception(
                "reco_df should not contain any user_item pairs that are already shown train_df"
            )

    def _get_all_user_item_pairs(self, df):
        return (
            df.select(self.user_col)
            .distinct()
            .join(df.select(self.item_col).distinct())
        )

    def _get_pairwise_items(self, df):

        return (
            df.select(self.user_col, F.col(self.item_col).alias("i1"))
            # get pairwise combinations of items per user (ignoring duplicate pairs [1,2] == [2,1])
            .join(
                df.select(
                    F.col(self.user_col).alias("_user"),
                    F.col(self.item_col).alias("i2"),
                ),
                (F.col(self.user_col) == F.col("_user")) & (F.col("i1") <= F.col("i2")),
            ).select(self.user_col, "i1", "i2")
        )

    def _get_cosine_similarity(self, n_partitions=200):
        if self.df_cosine_similariy is None:
            pairs = self._get_pairwise_items(df=self.train_df)
            item_count = self.train_df.groupBy(self.item_col).count()

            self.df_cosine_similariy = (
                pairs.groupBy("i1", "i2")
                .count()
                .join(
                    item_count.select(
                        F.col(self.item_col).alias("i1"),
                        F.pow(F.col("count"), 0.5).alias("i1_sqrt_count"),
                    ),
                    on="i1",
                )
                .join(
                    item_count.select(
                        F.col(self.item_col).alias("i2"),
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
                    ).alias("sim"),
                )
                .repartition(n_partitions, "i1", "i2")
                .sortWithinPartitions("i1", "i2")
            )
        return self.df_cosine_similariy

    # diversity metrics
    def _get_intralist_similarity(self, df):
        if self.df_intralist_similarity is None:
            pairs = self._get_pairwise_items(df=df)
            similarity_df = self._get_cosine_similarity().orderBy("i1", "i2")
            self.df_intralist_similarity = (
                pairs.join(similarity_df, on=["i1", "i2"], how="left")
                .fillna(
                    0
                )  # Fillna(0) is needed in the cases where similarity_df does not have an entry for a pair of items. e.g. i1 and i2 have never occurred together.
                .filter(F.col("i1") != F.col("i2"))
                .groupBy(self.user_col)
                .agg(F.mean(self.sim_col).alias("avg_il_sim"))
                .select(self.user_col, "avg_il_sim")
            )
        return self.df_intralist_similarity

    def user_diversity(self):
        """Calculate average diversity for recommendations for each user.

        Returns:
            pyspark.sql.dataframe.DataFrame: user_col, user_diversity
        """
        if self.df_user_diversity is None:
            self.df_intralist_similarity = self._get_intralist_similarity(self.reco_df)
            self.df_user_diversity = (
                self.df_intralist_similarity.withColumn(
                    "user_diversity", 1 - F.col("avg_il_sim")
                )
                .select(self.user_col, "user_diversity")
                .orderBy(self.user_col)
            )
        return self.df_user_diversity

    def diversity(self):
        """Calculate average diversity for recommendations across all users.

        Returns:
            pyspark.sql.dataframe.DataFrame: diversity
        """
        if self.df_diversity is None:
            self.df_user_diversity = self.user_diversity()
            self.df_diversity = self.df_user_diversity.select(
                F.mean("user_diversity").alias("diversity")
            )
        return self.df_diversity

    # novelty metrics
    def item_novelty(self):
        """Calculate novelty for each item in the recommendations.

        Returns:
            pyspark.sql.dataframe.DataFrame: item_col, item_novelty
        """
        if self.df_item_novelty is None:
            train_pairs = self._get_all_user_item_pairs(df=self.train_df)
            self.df_item_novelty = (
                train_pairs.join(
                    self.train_df.withColumn("seen", F.lit(1)),
                    on=[self.user_col, self.item_col],
                    how="left",
                )
                .filter(F.col("seen").isNull())
                .groupBy(self.item_col)
                .count()
                .join(
                    self.reco_df.groupBy(self.item_col).agg(
                        F.count(self.user_col).alias("reco_count")
                    ),
                    on=self.item_col,
                )
                .withColumn(
                    "item_novelty", -F.log2(F.col("reco_count") / F.col("count"))
                )
                .select(self.item_col, "item_novelty")
                .orderBy(self.item_col)
            )
        return self.df_item_novelty

    def user_novelty(self):
        """Calculate average item novelty for each user's recommendations.

        Returns:
            pyspark.sql.dataframe.DataFrame: user_col, user_novelty
        """
        if self.df_user_novelty is None:
            self.df_item_novelty = self.item_novelty()
            self.df_user_novelty = (
                self.reco_df.join(self.df_item_novelty, on=self.item_col)
                .groupBy(self.user_col)
                .agg(F.mean("item_novelty").alias("user_novelty"))
                .orderBy(self.user_col)
            )
        return self.df_user_novelty

    def novelty(self):
        """Calculate average novelty for recommendations across all users.

        Returns:
            pyspark.sql.dataframe.DataFrame: novelty
        """
        if self.df_novelty is None:
            self.df_user_novelty = self.user_novelty()
            self.df_novelty = self.df_user_novelty.agg(
                F.mean("user_novelty").alias("novelty")
            )
        return self.df_novelty

    # serendipity metrics
    def user_item_serendipity(self):
        """Calculate serendipity of each item in the recommendations for each user.

        Returns:
            pyspark.sql.dataframe.DataFrame: user_col, item_col, user_item_serendipity
        """
        # for every user_col, item_col in reco_df, join all interacted items from train_df.
        # These interacted items are repeated for each item in reco_df for a specific user.
        if self.df_user_item_serendipity is None:
            self.df_cosine_similariy = self._get_cosine_similarity().orderBy("i1", "i2")
            self.df_user_item_serendipity = (
                self.reco_df.withColumn(
                    "reco_item", F.col(self.item_col)
                )  # duplicate item_col to keep
                .select(
                    self.user_col,
                    "reco_item",
                    F.col(self.item_col).alias("reco_item_tmp"),
                )
                .join(
                    self.train_df.select(
                        self.user_col, F.col(self.item_col).alias("train_item_tmp")
                    ),
                    on=[self.user_col],
                )
                .select(
                    self.user_col,
                    "reco_item",
                    F.least(F.col("reco_item_tmp"), F.col("train_item_tmp")).alias(
                        "i1"
                    ),
                    F.greatest(F.col("reco_item_tmp"), F.col("train_item_tmp")).alias(
                        "i2"
                    ),
                )
                .join(self.df_cosine_similariy, on=["i1", "i2"], how="left")
                .fillna(0)
                .groupBy(self.user_col, F.col("reco_item").alias(self.item_col))
                .agg(F.mean(self.sim_col).alias("avg_item2interactedHistory_sim"))
                .join(self.reco_df, on=[self.user_col, self.item_col])
                .withColumn(
                    "user_item_serendipity",
                    (1 - F.col("avg_item2interactedHistory_sim"))
                    * F.col(self.relevance_col),
                )
                .select(self.user_col, self.item_col, "user_item_serendipity")
                .orderBy(self.user_col, self.item_col)
            )
        return self.df_user_item_serendipity

    def user_serendipity(self):
        """Calculate average serendipity for each user's recommendations.

        Returns:
            pyspark.sql.dataframe.DataFrame: user_col, user_serendipity
        """
        if self.df_user_serendipity is None:
            self.df_user_item_serendipity = self.user_item_serendipity()
            self.df_user_serendipity = (
                self.df_user_item_serendipity.groupBy(self.user_col)
                .agg(F.mean("user_item_serendipity").alias("user_serendipity"))
                .orderBy(self.user_col)
            )
        return self.df_user_serendipity

    def serendipity(self):
        """Calculate average serentipity for recommendations across all users.

        Returns:
            pyspark.sql.dataframe.DataFrame: serendipity
        """
        if self.df_serendipity is None:
            self.df_user_serendipity = self.user_serendipity()
            self.df_serendipity = self.df_user_serendipity.agg(
                F.mean("user_serendipity").alias("serendipity")
            )
        return self.df_serendipity

    # coverage metrics
    def catalog_coverage(self):
        """Calculate catalog coverage for recommendations across all users.

        Info:
            G. Shani and A. Gunawardana, Evaluating Recommendation Systems, Recommender Systems Handbook pp. 257-297, 2010.

        Returns:
            float: catalog coverage
        """
        # distinct item count in reco_df
        count_distinct_item_reco = self.reco_df.select(self.item_col).distinct().count()
        # distinct item count in train_df
        count_distinct_item_train = (
            self.train_df.select(self.item_col).distinct().count()
        )

        # cagalog coverage
        c_coverage = count_distinct_item_reco / count_distinct_item_train
        return c_coverage

    def distributional_coverage(self):
        """Calculate distributional coverage for recommendations across all users.

        Returns:
            float: distributional coverage
        """
        # In reco_df, how  many times each item_col is being recommended
        df_itemcnt_reco = self.reco_df.groupBy(self.item_col).count()
        # distinct item count in train_df
        count_distinct_item_train = (
            self.train_df.select(self.item_col).distinct().count()
        )
        # the number of total recommendations
        count_row_reco = self.reco_df.count()
        df_entropy = df_itemcnt_reco.withColumn(
            "p(i)", F.col("count") / count_row_reco
        ).withColumn("entropy(i)", F.col("p(i)") * F.log2(F.col("p(i)")))
        # distributional coverage
        d_coverage = (-2 / count_distinct_item_train) * df_entropy.agg(
            F.sum("entropy(i)")
        ).collect()[0][0]

        return d_coverage
