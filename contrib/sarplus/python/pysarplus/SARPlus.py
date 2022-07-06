# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""This is the implementation of SAR."""

import logging
import pandas as pd
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
)
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pysarplus import SARModel


SIM_COOCCUR = "cooccurrence"
SIM_JACCARD = "jaccard"
SIM_LIFT = "lift"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("sarplus")


class SARPlus:
    """SAR implementation for PySpark."""

    def __init__(
        self,
        spark,
        col_user="userID",
        col_item="itemID",
        col_rating="rating",
        col_timestamp="timestamp",
        table_prefix="",
        similarity_type="jaccard",
        time_decay_coefficient=30,
        time_now=None,
        timedecay_formula=False,
        threshold=1,
        cache_path=None,
    ):

        """Initialize model parameters
        Args:
            spark (pyspark.sql.SparkSession): Spark session
            col_user (str): user column name
            col_item (str): item column name
            col_rating (str): rating column name
            col_timestamp (str): timestamp column name
            table_prefix (str): name prefix of the generated tables
            similarity_type (str): ['cooccurrence', 'jaccard', 'lift']
                option for computing item-item similarity
            time_decay_coefficient (float): number of days till
                ratings are decayed by 1/2.  denominator in time
                decay.  Zero makes time decay irrelevant
            time_now (int | None): current time for time decay
                calculation
            timedecay_formula (bool): flag to apply time decay
            threshold (int): item-item co-occurrences below this
                threshold will be removed
            cache_path (str): user specified local cache directory for
                recommend_k_items().  If specified,
                recommend_k_items() will do C++ based fast
                predictions.
        """
        assert threshold > 0

        self.spark = spark
        self.header = {
            "col_user": col_user,
            "col_item": col_item,
            "col_rating": col_rating,
            "col_timestamp": col_timestamp,
            "prefix": table_prefix,
            "time_now": time_now,
            "time_decay_half_life": time_decay_coefficient * 24 * 60 * 60,
            "threshold": threshold,
        }
        if similarity_type not in [SIM_COOCCUR, SIM_JACCARD, SIM_LIFT]:
            raise ValueError(
                'Similarity type must be one of ["cooccurrence" | "jaccard" | "lift"]'
            )
        self.similarity_type = similarity_type
        self.timedecay_formula = timedecay_formula
        self.item_similarity = None
        self.cache_path = cache_path

    def _format(self, string, **kwargs):
        return string.format(**self.header, **kwargs)

    def fit(self, df):
        """Main fit method for SAR.

        Expects the dataframes to have row_id, col_id columns which
        are indexes, i.e. contain the sequential integer index of the
        original alphanumeric user and item IDs.  Dataframe also
        contains rating and timestamp as floats; timestamp is in
        seconds since Epoch by default.

        Arguments:
            df (pySpark.DataFrame): input dataframe which contains the
                index of users and items.
        """

        df.createOrReplaceTempView(self._format("{prefix}df_train_input"))

        if self.timedecay_formula:
            # With time decay, we compute a sum over ratings given by
            # a user in the case when T=np.inf, so user gets a
            # cumulative sum of ratings for a particular item and not
            # the last rating.  Time Decay does a group by on user
            # item pairs and apply the formula for time decay there
            # Time T parameter is in days and input time is in
            # seconds, so we do dt/60/(T*24*60)=dt/(T*24*3600) the
            # following is the query which we want to run

            if self.header["time_now"] is None:
                query = self._format("""
                    SELECT CAST(MAX(`{col_timestamp}`) AS long)
                    FROM `{prefix}df_train_input`
                """)
                self.header["time_now"] = self.spark.sql(query).first()[0]

            query = self._format("""
                SELECT `{col_user}`,
                       `{col_item}`,
                       SUM(
                           `{col_rating}` *
                           POW(2, (CAST(`{col_timestamp}` AS LONG) - {time_now}) / {time_decay_half_life})
                          ) AS `{col_rating}`
                FROM `{prefix}df_train_input`
                GROUP BY `{col_user}`, `{col_item}`
                CLUSTER BY `{col_user}`
            """)

            # replace with time-decayed version
            df = self.spark.sql(query)
        else:
            # we need to de-duplicate items by using the latest item
            query = self._format("""
                SELECT `{col_user}`, `{col_item}`, `{col_rating}`
                FROM (
                      SELECT `{col_user}`,
                             `{col_item}`,
                             `{col_rating}`,
                             ROW_NUMBER() OVER user_item_win AS latest
                      FROM `{prefix}df_train_input`
                     ) AS reverse_chrono_table
                WHERE reverse_chrono_table.latest = 1
                WINDOW user_item_win AS (
                    PARTITION BY `{col_user}`,`{col_item}`
                    ORDER BY `{col_timestamp}` DESC)
            """)

            df = self.spark.sql(query)

        df.createOrReplaceTempView(self._format("{prefix}df_train"))

        log.info("sarplus.fit 1/2: compute item cooccurrences...")

        # compute cooccurrence above minimum threshold
        query = self._format("""
            SELECT a.`{col_item}` AS i1,
                   b.`{col_item}` AS i2,
                   COUNT(*) AS value
            FROM `{prefix}df_train` AS a
            INNER JOIN `{prefix}df_train` AS b
            ON a.`{col_user}` = b.`{col_user}` AND a.`{col_item}` <= b.`{col_item}`
            GROUP BY i1, i2
            HAVING value >= {threshold}
            CLUSTER BY i1, i2
        """)

        item_cooccurrence = self.spark.sql(query)
        item_cooccurrence.write.mode("overwrite").saveAsTable(
            self._format("{prefix}item_cooccurrence")
        )

        # compute the diagonal used later for Jaccard and Lift
        if self.similarity_type == SIM_LIFT or self.similarity_type == SIM_JACCARD:
            query = self._format("""
                SELECT i1 AS i, value AS margin
                FROM `{prefix}item_cooccurrence`
                WHERE i1 = i2
            """)
            item_marginal = self.spark.sql(query)
            item_marginal.createOrReplaceTempView(self._format("{prefix}item_marginal"))

        if self.similarity_type == SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif self.similarity_type == SIM_JACCARD:
            query = self._format("""
                SELECT i1, i2, value / (m1.margin + m2.margin - value) AS value
                FROM `{prefix}item_cooccurrence` AS a
                INNER JOIN `{prefix}item_marginal` AS m1 ON a.i1 = m1.i
                INNER JOIN `{prefix}item_marginal` AS m2 ON a.i2 = m2.i
                CLUSTER BY i1, i2
            """)
            self.item_similarity = self.spark.sql(query)
        elif self.similarity_type == SIM_LIFT:
            query = self._format("""
                SELECT i1, i2, value / (m1.margin * m2.margin) AS value
                FROM `{prefix}item_cooccurrence` AS a
                INNER JOIN `{prefix}item_marginal` AS m1 ON a.i1 = m1.i
                INNER JOIN `{prefix}item_marginal` AS m2 ON a.i2 = m2.i
                CLUSTER BY i1, i2
            """)
            self.item_similarity = self.spark.sql(query)
        else:
            raise ValueError(
                "Unknown similarity type: {0}".format(self.similarity_type)
            )

        # store upper triangular
        log.info(
            "sarplus.fit 2/2: compute similarity metric %s..." % self.similarity_type
        )
        self.item_similarity.write.mode("overwrite").saveAsTable(
            self._format("{prefix}item_similarity_upper")
        )

        # expand upper triangular to full matrix

        query = self._format("""
            SELECT i1, i2, value
            FROM (
                  (
                   SELECT i1, i2, value
                   FROM `{prefix}item_similarity_upper`
                  )
                  UNION ALL
                  (
                   SELECT i2 AS i1, i1 AS i2, value
                   FROM `{prefix}item_similarity_upper`
                   WHERE i1 <> i2
                  )
                 )
            CLUSTER BY i1
        """)

        self.item_similarity = self.spark.sql(query)
        self.item_similarity.write.mode("overwrite").saveAsTable(
            self._format("{prefix}item_similarity")
        )

        # free space
        self.spark.sql(self._format("DROP TABLE `{prefix}item_cooccurrence`"))
        self.spark.sql(self._format("DROP TABLE `{prefix}item_similarity_upper`"))

        self.item_similarity = self.spark.table(
            self._format("{prefix}item_similarity")
        )

    def get_user_affinity(self, test):
        """Prepare test set for C++ SAR prediction code.
        Find all items the test users have seen in the past.

        Arguments:
            test (pySpark.DataFrame): input dataframe which contains test users.
        """
        test.createOrReplaceTempView(self._format("{prefix}df_test"))

        query = self._format("""
            SELECT DISTINCT `{col_user}`
            FROM `{prefix}df_test`
            CLUSTER BY `{col_user}`
        """)

        df_test_users = self.spark.sql(query)
        df_test_users.write.mode("overwrite").saveAsTable(
            self._format("{prefix}df_test_users")
        )

        query = self._format("""
            SELECT a.`{col_user}`,
                   a.`{col_item}`,
                   CAST(a.`{col_rating}` AS double) AS `{col_rating}`
            FROM `{prefix}df_train` AS a
            INNER JOIN `{prefix}df_test_users` AS b
            ON a.`{col_user}` = b.`{col_user}`
            DISTRIBUTE BY `{col_user}`
            SORT BY `{col_user}`, `{col_item}`
        """)

        return self.spark.sql(query)

    def _recommend_k_items_fast(
        self,
        test,
        top_k=10,
        remove_seen=True,
        n_user_prediction_partitions=200,
    ):

        assert self.cache_path is not None

        # create item id to continuous index mapping
        log.info("sarplus.recommend_k_items 1/3: create item index")
        query = self._format("""
            SELECT i1, ROW_NUMBER() OVER(ORDER BY i1)-1 AS idx
            FROM (
                  SELECT DISTINCT i1
                  FROM `{prefix}item_similarity`
                 )
            CLUSTER BY i1
        """)
        self.spark.sql(query).write.mode("overwrite").saveAsTable(
            self._format("{prefix}item_mapping")
        )

        # map similarity matrix into index space
        query = self._format("""
            SELECT a.idx AS i1, b.idx AS i2, is.value
            FROM `{prefix}item_similarity` AS is,
                 `{prefix}item_mapping` AS a,
                 `{prefix}item_mapping` AS b
            WHERE is.i1 = a.i1 AND i2 = b.i1
        """)
        self.spark.sql(query).write.mode("overwrite").saveAsTable(
            self._format("{prefix}item_similarity_mapped")
        )

        cache_path_output = self.cache_path
        if self.cache_path.startswith("dbfs:"):
            # Databricks DBFS
            cache_path_input = "/dbfs" + self.cache_path[5:]
        elif self.cache_path.startswith("synfs:"):
            # Azure Synapse
            # See https://docs.microsoft.com/en-us/azure/synapse-analytics/spark/synapse-file-mount-api
            cache_path_input = "/synfs" + self.cache_path[6:]
        else:
            cache_path_input = self.cache_path

        # export similarity matrix for C++ backed UDF
        log.info("sarplus.recommend_k_items 2/3: prepare similarity matrix")

        query = self._format("""
            SELECT i1, i2, CAST(value AS DOUBLE) AS value
            FROM `{prefix}item_similarity_mapped`
            ORDER BY i1, i2
        """)
        self.spark.sql(query).coalesce(1).write.format(
            "com.microsoft.sarplus"
        ).mode("overwrite").save(cache_path_output)

        self.get_user_affinity(test).createOrReplaceTempView(
            self._format("{prefix}user_affinity")
        )

        # map item ids to index space
        query = self._format("""
            SELECT `{col_user}`, idx, rating
            FROM (
                  SELECT `{col_user}`, b.idx, `{col_rating}` AS rating
                  FROM `{prefix}user_affinity`
                  JOIN `{prefix}item_mapping` AS b
                  ON `{col_item}` = b.i1 
                 )
            CLUSTER BY `{col_user}`
        """)
        pred_input = self.spark.sql(query)

        schema = StructType(
            [
                StructField(
                    "userID",
                    pred_input.schema[self.header["col_user"]].dataType,
                    True
                ),
                StructField("itemID", IntegerType(), True),
                StructField("score", FloatType(), True),
            ]
        )

        # make sure only the header is pickled
        local_header = self.header

        # bridge to python/C++
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def sar_predict_udf(df):
            # Magic happening here.  The cache_path points to file write
            # to by com.microsoft.sarplus This has exactly the memory
            # layout we need and since the file is memory mapped, the
            # memory consumption only happens once per worker for all
            # python processes
            model = SARModel(cache_path_input)
            preds = model.predict(
                df["idx"].values, df["rating"].values, top_k, remove_seen
            )

            user = df[local_header["col_user"]].iloc[0]

            preds_ret = pd.DataFrame(
                [(user, x.id, x.score) for x in preds], columns=range(3)
            )

            return preds_ret

        log.info("sarplus.recommend_k_items 3/3: compute recommendations")

        df_preds = (
            pred_input.repartition(
                n_user_prediction_partitions, self.header["col_user"]
            )
            .groupby(self.header["col_user"])
            .apply(sar_predict_udf)
        )

        df_preds.createOrReplaceTempView(self._format("{prefix}predictions"))

        query = self._format("""
            SELECT userID AS `{col_user}`, b.i1 AS `{col_item}`, score
            FROM `{prefix}predictions` AS p, `{prefix}item_mapping` AS b
            WHERE p.itemID = b.idx
        """)
        return self.spark.sql(query)

    def _recommend_k_items_slow(self, test, top_k=10, remove_seen=True):
        """Recommend top K items for all users which are in the test set.

        Args:
            test: test Spark dataframe
            top_k: top n items to return
            remove_seen: remove items test users have already seen in
                the past from the recommended set.
        """

        # TODO: remove seen
        if remove_seen:
            raise ValueError("Not implemented")

        self.get_user_affinity(test).write.mode("overwrite").saveAsTable(
            self._format("{prefix}user_affinity")
        )

        # user_affinity * item_similarity
        # filter top-k
        query = self._format("""
            SELECT `{col_user}`, `{col_item}`, score
            FROM (
                  SELECT df.`{col_user}`,
                         s.i2 AS `{col_item}`,
                         SUM(df.`{col_rating}` * s.value) AS score,
                         ROW_NUMBER() OVER w AS rank
                  FROM `{prefix}user_affinity` AS df,
                       `{prefix}item_similarity` AS s
                  WHERE df.`{col_item}` = s.i1
                  WINDOW w AS (
                      PARTITION BY `{col_user}`
                      ORDER BY SUM(df.`{col_rating}` * s.value) DESC)
                  GROUP BY df.`{col_user}`, s.i2
                 )
            WHERE rank <= {top_k}
        """, top_k=top_k)

        return self.spark.sql(query)

    def recommend_k_items(
        self,
        test,
        top_k=10,
        remove_seen=True,
        use_cache=False,
        n_user_prediction_partitions=200,
    ):
        """Recommend top K items for all users which are in the test set.

        Args:
            test (pyspark.sql.DataFrame): test Spark dataframe.
            top_k (int): top n items to return.
            remove_seen (bool): remove items test users have already
                seen in the past from the recommended set.
            use_cache (bool): use specified local directory stored in
                `self.cache_path` as cache for C++ based fast
                predictions.
            n_user_prediction_partitions (int): prediction partitions.

        Returns:
            pyspark.sql.DataFrame: Spark dataframe with recommended items
        """
        if not use_cache:
            return self._recommend_k_items_slow(test, top_k, remove_seen)
        elif self.cache_path is not None:
            return self._recommend_k_items_fast(
                test, top_k, remove_seen, n_user_prediction_partitions
            )
        else:
            raise ValueError("No cache_path specified")
