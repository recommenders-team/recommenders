# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pyspark.sql.types import *
from pyspark.sql import functions as F

from reco_utils.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
)


class DiversityEvaluation:
    def __init__(
        self,
        train_df,
        reco_df,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_relevance=None,
    ):
        """Diversity evaluator.
        train (train_df) and recommendation (reco_df) dataframes should already be groupped by user-item pair.

        Metrics include:
            Coverage - The proportion of items that can be recommended. It includes two metrics: (1) catalog_coverage, which measures the proportion of items that get recommended from the item catalog; (2) distributional_coverage, which measures how unequally different items are recommended in the recommendations to all users.
            Novelty - A more novel item indicates it is less popular, i.e., it gets recommended less frequently.
            Diversity - The dissimilarity of items being recommended.
            Serendipity - The "unusualness" or "surprise" of recommendations to a user. When 'col_relevance' is used, it indicates how "pleasant surprise" of recommendations is to a user. 


        Args:
            train_df (pySpark DataFrame): Training set used for the recommender,
                containing col_user, col_item.
            reco_df (pySpark DataFrame): Recommender's prediction output,
                containing col_user, col_item, col_relevance (optional).
            col_user (str): User id column name.
            col_item (str): Item id column name.
            col_relevance (str): This column indicates whether the recommended item is actually relevant to the user or not.
        """

        self.train_df = train_df.select(col_user, col_item)
        self.col_user = col_user
        self.col_item = col_item
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

        if col_relevance is None:
            self.col_relevance = "relevance"
            # relevance term, default is 1 (relevent) for all
            self.reco_df = reco_df.select(
                col_user, col_item, F.lit(1.0).alias(self.col_relevance)
            )
        else:
            self.col_relevance = col_relevance
            self.reco_df = reco_df.select(
                col_user, col_item, F.col(self.col_relevance).cast(DoubleType())
            )

        # check if reco_df contain any user_item pairs that are already shown train_df
        count_intersection = (
            self.train_df.select(self.col_user, self.col_item)
            .intersect(self.reco_df.select(self.col_user, self.col_item))
            .count()
        )

        if count_intersection != 0:
            raise Exception(
                "reco_df should not contain any user_item pairs that are already shown train_df"
            )

    def _get_all_user_item_pairs(self, df):
        return (
            df.select(self.col_user)
            .distinct()
            .join(df.select(self.col_item).distinct())
        )

    def _get_pairwise_items(self, df):
        return (
            df.select(self.col_user, F.col(self.col_item).alias("i1"))
            # get pairwise combinations of items per user (ignoring duplicate pairs [1,2] == [2,1])
            .join(
                df.select(
                    F.col(self.col_user).alias("_user"),
                    F.col(self.col_item).alias("i2"),
                ),
                (F.col(self.col_user) == F.col("_user")) & (F.col("i1") <= F.col("i2")),
            ).select(self.col_user, "i1", "i2")
        )

    def _get_cosine_similarity(self, n_partitions=200):
        if self.df_cosine_similariy is None:
            pairs = self._get_pairwise_items(df=self.train_df)
            item_count = self.train_df.groupBy(self.col_item).count()

            self.df_cosine_similariy = (
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
                .groupBy(self.col_user)
                .agg(F.mean(self.sim_col).alias("avg_il_sim"))
                .select(self.col_user, "avg_il_sim")
            )
        return self.df_intralist_similarity

    def user_diversity(self):
        """Calculate average diversity for recommendations for each user.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: col_user, user_diversity.
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
        """Calculate average diversity for recommendations across all users.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: diversity.
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
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: col_item, item_novelty.
        """
        if self.df_item_novelty is None:
            train_pairs = self._get_all_user_item_pairs(df=self.train_df)
            self.df_item_novelty = (
                train_pairs.join(
                    self.train_df.withColumn("seen", F.lit(1)),
                    on=[self.col_user, self.col_item],
                    how="left",
                )
                .filter(F.col("seen").isNull())
                .groupBy(self.col_item)
                .count()
                .join(
                    self.reco_df.groupBy(self.col_item).agg(
                        F.count(self.col_user).alias("reco_count")
                    ),
                    on=self.col_item,
                )
                .withColumn(
                    "item_novelty", -F.log2(F.col("reco_count") / F.col("count"))
                )
                .select(self.col_item, "item_novelty")
                .orderBy(self.col_item)
            )
        return self.df_item_novelty

    def user_novelty(self):
        """Calculate average item novelty for each user's recommendations.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: col_user, user_novelty.
        """
        if self.df_user_novelty is None:
            self.df_item_novelty = self.item_novelty()
            self.df_user_novelty = (
                self.reco_df.join(self.df_item_novelty, on=self.col_item)
                .groupBy(self.col_user)
                .agg(F.mean("item_novelty").alias("user_novelty"))
                .orderBy(self.col_user)
            )
        return self.df_user_novelty

    def novelty(self):
        """Calculate average novelty for recommendations across all users.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: novelty.
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
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: col_user, col_item, user_item_serendipity.
        """
        # for every col_user, col_item in reco_df, join all interacted items from train_df.
        # These interacted items are repeated for each item in reco_df for a specific user.
        if self.df_user_item_serendipity is None:
            self.df_cosine_similariy = self._get_cosine_similarity().orderBy("i1", "i2")
            self.df_user_item_serendipity = (
                self.reco_df.select(
                    self.col_user,
                    self.col_item,
                    F.col(self.col_item).alias("reco_item_tmp"),  # duplicate col_item to keep
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
                .join(self.df_cosine_similariy, on=["i1", "i2"], how="left")
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
        """Calculate average serentipity for recommendations across all users.

        Returns:
            pyspark.sql.dataframe.DataFrame: A dataframe with following columns: serendipity.
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
        count_distinct_item_reco = self.reco_df.select(self.col_item).distinct().count()
        # distinct item count in train_df
        count_distinct_item_train = (
            self.train_df.select(self.col_item).distinct().count()
        )

        # cagalog coverage
        c_coverage = count_distinct_item_reco / count_distinct_item_train
        return c_coverage

    def distributional_coverage(self):
        """Calculate distributional coverage for recommendations across all users.

        Returns:
            float: distributional coverage
        """
        # In reco_df, how  many times each col_item is being recommended
        df_itemcnt_reco = self.reco_df.groupBy(self.col_item).count()
        # distinct item count in train_df
        count_distinct_item_train = (
            self.train_df.select(self.col_item).distinct().count()
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
