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
        self.df_user_item_serendipity = None
        self.df_item_novelty = None
        self.df_intralist_similarity = None

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

    def get_all_user_item_pairs(self, df):
        return (
            df.select(self.user_col)
            .distinct()
            .join(df.select(self.item_col).distinct())
        )

    def get_pairwise_items(self, df, full_matrix=False):
        if full_matrix == False:
            return (
                df.select(self.user_col, F.col(self.item_col).alias("i1"))
                # get pairwise combinations of items per user (ignoring duplicate pairs [1,2] == [2,1])
                .join(
                    df.select(
                        F.col(self.user_col).alias("_user"),
                        F.col(self.item_col).alias("i2"),
                    ),
                    (F.col(self.user_col) == F.col("_user"))
                    & (F.col("i1") <= F.col("i2")),
                ).select(self.user_col, "i1", "i2")
            )
        else:
            return (
                df.select(self.user_col, F.col(self.item_col).alias("i1"))
                # get pairwise combinations of items per user (including both pairs [1,2] and [2,1])
                .join(
                    df.select(
                        F.col(self.user_col).alias("_user"),
                        F.col(self.item_col).alias("i2"),
                    ),
                    (F.col(self.user_col) == F.col("_user")),
                ).select(self.user_col, "i1", "i2")
            )

    def get_cosine_similarity(self, full_matrix=False, n_partitions=200):
        # TODO: make sure there are no null values in user or item columns
        # TODO: make sure temporary column names don't match existing user or item column names

        pairs = self.get_pairwise_items(df=self.train_df, full_matrix=full_matrix)
        item_count = self.train_df.groupBy(self.item_col).count()

        return (
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
                    F.col("count") / (F.col("i1_sqrt_count") * F.col("i2_sqrt_count"))
                ).alias("sim"),
            )
            .repartition(n_partitions, "i1", "i2")
            .sortWithinPartitions("i1", "i2")
        )

    # diversity metrics
    def get_intralist_similarity(self, df, similarity_df):
        pairs = self.get_pairwise_items(df=df)
        return (
            pairs.join(similarity_df, on=["i1", "i2"], how="left")
            .fillna(
                0
            )  # Fillna(0) is needed in the cases where similarity_df does not have an entry for a pair of items. e.g. i1 and i2 have never occurred together.
            .filter(F.col("i1") != F.col("i2"))
            .groupBy(self.user_col)
            .agg(F.mean(self.sim_col).alias("avg_il_sim"))
            .select(self.user_col, "avg_il_sim")
        )

    def user_diversity(self):
        if self.df_intralist_similarity is None:
            cossim = self.get_cosine_similarity().orderBy("i1", "i2")
            self.df_intralist_similarity = self.get_intralist_similarity(
                df=self.reco_df, similarity_df=cossim
            )
        return (
            self.df_intralist_similarity.withColumn(
                "diversity", 1 - F.col("avg_il_sim")
            )
            .select(self.user_col, "diversity")
            .orderBy(self.user_col)
        )

    def diversity(self):
        # TODO: add error handling logic for conditions where user_id is not valid
        if self.df_intralist_similarity is None:
            cossim = self.get_cosine_similarity().orderBy("i1", "i2")
            self.df_intralist_similarity = self.get_intralist_similarity(
                df=self.reco_df, similarity_df=cossim
            )
        return self.df_intralist_similarity.withColumn(
            "diversity", 1 - F.col("avg_il_sim")
        ).select(F.mean("diversity").alias("diversity"))

    # novelty metrics
    def get_item_novelty(self):
        train_pairs = self.get_all_user_item_pairs(df=self.train_df)
        return (
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
            .withColumn("item_novelty", -F.log2(F.col("reco_count") / F.col("count")))
        )

    def item_novelty(self):
        if self.df_item_novelty is None:
            self.df_item_novelty = self.get_item_novelty()
        return self.df_item_novelty.select(self.item_col, "item_novelty").orderBy(
            self.item_col
        )

    def user_novelty(self):
        if self.df_item_novelty is None:
            self.df_item_novelty = self.get_item_novelty()
        return (
            self.reco_df.join(self.df_item_novelty, on=self.item_col)
            .groupBy(self.user_col)
            .agg(F.mean("item_novelty").alias("user_novelty"))
            .orderBy(self.user_col)
        )

    def novelty(self):
        # TODO: add error handling logic for any other conditions
        if self.df_item_novelty is None:
            self.df_item_novelty = self.get_item_novelty()
        return self.reco_df.join(self.df_item_novelty, on=self.item_col).agg(
            F.mean("item_novelty").alias("novelty")
        )

    # serendipity metrics
    def get_user_item_serendipity(self):
        # TODO: add relevance term as input parameter

        # for every user_col, item_col in reco_df, join all interacted items from train_df.
        # These interacted items are reapeated for each item in reco_df for a specific user.
        reco_item_interacted_history = (
            self.reco_df.withColumn("i1", F.col(self.item_col))
            .join(
                self.train_df.withColumn("i2", F.col(self.item_col)), on=[self.user_col]
            )
            .select(self.user_col, "i1", "i2")
        )
        cossim_full = self.get_cosine_similarity(full_matrix=True).orderBy("i1", "i2")
        join_sim = (
            reco_item_interacted_history.join(cossim_full, on=["i1", "i2"], how="left")
            .fillna(0)
            .groupBy(self.user_col, "i1")
            .agg(F.mean(self.sim_col).alias("avg_item2interactedHistory_sim"))
            .withColumn(self.item_col, F.col("i1"))
            .drop("i1")
        )
        return join_sim.join(
            self.reco_df, on=[self.user_col, self.item_col]
        ).withColumn(
            "user_item_serendipity",
            (1 - F.col("avg_item2interactedHistory_sim")) * F.col(self.relevance_col),
        )

    def user_item_serendipity(self):
        if self.df_user_item_serendipity is None:
            self.df_user_item_serendipity = self.get_user_item_serendipity()

        return self.df_user_item_serendipity.select(
            self.user_col, self.item_col, "user_item_serendipity"
        ).orderBy(self.user_col, self.item_col)

    def user_serendipity(self):
        if self.df_user_item_serendipity is None:
            self.df_user_item_serendipity = self.get_user_item_serendipity()

        return (
            self.df_user_item_serendipity.groupBy(self.user_col)
            .agg(F.mean("user_item_serendipity").alias("user_serendipity"))
            .orderBy(self.user_col)
        )

    def serendipity(self):
        # TODO: add error handling logic for any other conditions
        if self.df_user_item_serendipity is None:
            self.df_user_item_serendipity = self.get_user_item_serendipity()

        return self.df_user_item_serendipity.agg(
            F.mean("user_item_serendipity").alias("serendipity")
        )

    # coverage metrics
    def catalog_coverage(self):
        # distinct item count in reco_df
        count_distinct_item_reco = self.reco_df.select(
            F.countDistinct(self.item_col)
        ).collect()[0][0]
        # distinct item count in train_df
        count_distinct_item_train = self.train_df.select(
            F.countDistinct(self.item_col)
        ).collect()[0][0]

        # cagalog coverage
        c_coverage = count_distinct_item_reco / count_distinct_item_train
        return c_coverage

    def distributional_coverage(self):
        # In reco_df, how  many times each item_col is being recommended
        df_itemcnt_reco = self.reco_df.groupBy(self.item_col).count()
        # distinct item count in train_df
        count_distinct_item_train = self.train_df.select(
            F.countDistinct(self.item_col)
        ).collect()[0][0]
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
