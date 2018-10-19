"""
Reference implementation of SAR in pySpark using Spark-SQL and some dataframe operations.
This is supposed to be a super-performant implementation of SAR on Spark using pySpark.

PS: there is plenty of room for improvement, especially around the very last step of making a partial sort:
1) Can be done using UDAFs
2) UDAFs can be transformed into: pivot, series of UDF operations, pivot
3) other DF operations.
"""

import numpy as np
import logging
import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, create_map, sum
from pyspark.sql.window import Window
from itertools import chain

from utilities.common.constants import (
    DEFAULT_USER_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    TIMESTAMP_COL,
)
from utilities.common.constants import PREDICTION_COL

from utilities.recommender.sar import (
    SIM_JACCARD,
    SIM_LIFT,
    SIM_COOCCUR,
    HASHED_USERS,
    HASHED_ITEMS,
)
from utilities.recommender.sar import (
    TIME_DECAY_COEFFICIENT,
    TIME_NOW,
    TIMEDECAY_FORMULA,
    THRESHOLD,
)

"""
enable or set manually with --log=INFO when running example file if you want logging:
disabling because logging output contaminates stdout output on Databricsk Spark clusters
"""
# logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class SARpySparkReference:
    """SAR reference implementation"""

    def __init__(
        self,
        spark,
        remove_seen=True,
        col_user=DEFAULT_USER_COL,
        col_item=DEFAULT_ITEM_COL,
        col_rating=DEFAULT_RATING_COL,
        col_timestamp=TIMESTAMP_COL,
        similarity_type=SIM_JACCARD,
        time_decay_coefficient=TIME_DECAY_COEFFICIENT,
        time_now=TIME_NOW,
        timedecay_formula=TIMEDECAY_FORMULA,
        threshold=THRESHOLD,
        debug=False,
    ):

        self.col_rating = col_rating
        self.col_item = col_item
        self.col_user = col_user
        # default values for all SAR algos
        self.col_timestamp = col_timestamp

        self.remove_seen = remove_seen

        # time of item-item similarity
        self.similarity_type = similarity_type
        # denominator in time decay. Zero makes time decay irrelevant
        self.time_decay_coefficient = time_decay_coefficient
        # toggle the computation of time decay group by formula
        self.timedecay_formula = timedecay_formula
        # current time for time decay calculation
        self.time_now = time_now
        # cooccurrence matrix threshold
        self.threshold = threshold
        # debug the code
        self.debug = debug
        # log the length of operations
        self.timer_log = []

        # array of indexes for rows and columns of users and items in training set
        self.index = None
        self.model_str = "sar_pyspark"
        self.model = self
        # spark context
        self.spark = spark

        # we use these handles for unit tests
        self.item_similarity = None
        self.affinity = None

        # threshold - items below this number get set to zero in coocurrence counts
        assert self.threshold > 0

        # more columns which are used internally
        self._col_hashed_items = HASHED_ITEMS
        self._col_hashed_users = HASHED_USERS

        # Obtain all the users and items from both training and test data
        self.unique_users = None
        self.unique_items = None
        # store training set index for future use during prediction
        self.index = None

        # user2rowID map for prediction method to look up user affinity vectors
        self.user_map_dict = None
        # mapping for item to matrix element
        self.item_map_dict = None

        # the opposite of the above map - map array index to actual string ID
        self.index2user = None
        self.index2item = None

        # affinity scores for the recommendation
        self.scores = None

        # indexed IDs of users which are in the training set
        self.user_row_ids = None

        # training dataframe reference
        self.df = None

    def set_index(
        self,
        unique_users,
        unique_items,
        user_map_dict,
        item_map_dict,
        index2user,
        index2item,
    ):
        """MVP2 temporary function to set the index of the sparse dataframe.
        In future releases this will be carried out into the data object and index will be provided
        with the data"""

        # original IDs of users and items in a list
        # later as we modify the algorithm these might not be needed (can use dictionary keys
        # instead)
        self.unique_users = unique_users
        self.unique_items = unique_items

        # mapping of original IDs to actual matrix elements
        self.user_map_dict = user_map_dict
        self.item_map_dict = item_map_dict

        # reverse mapping of matrix index to an item
        # TODO: we can make this into an array as well
        self.index2user = index2user
        self.index2item = index2item

    def get_item_similarity_as_matrix(self):
        """Used for unit tests only - write the SQL table as a smaller numpy array.

        Returns
            np.array: item similarity as matrix."""

        if self.item_similarity is None or self.item_similarity.count() == 0:
            return None

        isp = self.item_similarity.toPandas()
        # adjust index for matrices
        isp["row_item_id"] = isp["row_item_id"] - 1
        isp["col_item_id"] = isp["col_item_id"] - 1

        assert isp["row_item_id"].max() == isp["col_item_id"].max()
        matrix = np.zeros((len(self.unique_items), len(self.unique_items)))
        matrix[isp.row_item_id.tolist(), isp.col_item_id.tolist()] = isp.value.tolist()
        return matrix

    def get_user_affinity_as_vector(self, uid):
        """Returns a numpy array vector of user affnity values for a particular user.

        Args:
            uid (str/int): actual ID of the user (not the index)

        Returns:
            np.array: 1D array of user affinities."""

        if self.affinity is None or self.affinity.count() == 0:
            return None

        row_id = self.user_map_dict[uid]

        self.affinity.createOrReplaceTempView("affinity")
        ap = self.spark.sql(
            "select * from affinity where row_id = %d" % row_id
        ).toPandas()

        ap["col_id"] = ap["col_id"] - 1
        n = ap["col_id"].max()
        affinity_vector = np.zeros((1, len(self.unique_items)))
        affinity_vector[0, ap.col_id.tolist()] = ap.Affinity.tolist()

        return affinity_vector

    def fit(self, df):
        """Main fit method for SAR. Expects the dataframes to have row_id, col_id columns which are indexes,
        i.e. contain the sequential integer index of the original alphanumeric user and item IDs.
        Dataframe also contains rating and timestamp as floats; timestamp is in seconds since Epoch by default.

        Arguments:
            df (pySpark.DataFrame): input dataframe which contains the index of users and items. """

        # record the training dataframe
        self.df = df
        df.createOrReplaceTempView("df_train")

        # record all user IDs in the training set
        # self.user_row_ids = [df_trainx[0] for x in self.spark.sql("select distinct row_id from df_train").collect()]
        self.user_row_ids = [x[0] for x in df.select("row_id").distinct().collect()]

        log.info("Collecting user affinity matrix...")

        if self.timedecay_formula:
            # WARNING: previously we would take the last value in training dataframe and set it
            # as a matrix U element
            # for each user-item pair. Now with time decay, we compute a sum over ratings given
            # by a user in the case
            # when T=np.inf, so user gets a cumulative sum of ratings for a particular item and
            # not the last rating.
            log.info("Calculating time-decayed affinities...")
            # Time Decay
            # do a group by on user item pairs and apply the formula for time decay there
            # Time T parameter is in days and input time is in seconds
            # so we do dt/60/(T*24*60)=dt/(T*24*3600)
            # the folling is the query which we want to run
            if self.time_now is None:
                self.time_now = df.select(F.max(self.col_timestamp)).first()[0]

            """
            select
            row_id, col_id, sum(Rating * exp(-log(2) * (t0 - Timestamp) / (T * 3600 * 24))) as Affinity
            from df_train group
            by
            row_id, col_id
            """
            query = """select
            row_id, col_id, sum(%s * exp(-log(2) * (%f - %s) / (%f * 3600 * 24))) as Affinity
            from df_train group
            by
            row_id, col_id""" % (
                self.col_rating,
                self.time_now,
                self.col_timestamp,
                self.time_decay_coefficient,
            )
            log.info("Running query -- " + query)
            df = self.spark.sql(query)

        else:
            # without time decay we take the last user-provided rating supplied in the dataset as the
            # final rating for the user-item pair
            logging.info("Deduplicating the user-item counts")
            query = (
                "select distinct row_id, col_id, "
                + self.col_rating
                + " as Affinity from df_train"
            )
            log.info("Running query -- " + query)
            df = self.spark.sql(query)

        # record affinity scores
        self.affinity = df
        if self.debug:
            # trigger execution
            self.time()
            cnt = self.affinity.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Affinity calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        # create affinity transpose
        log.info("Calculating item cooccurrence...")

        # Calculate item cooccurrence by computing:
        #  C = U'.transpose() * U'
        # where U' is the user_affinity matrix with 1's as values (instead of ratings)

        # add dummy indicator column for DF iterables
        df = df.withColumn("ind", lit(1))
        df_transpose = df.select(
            col("col_id").alias("row_id"),
            col("row_id").alias("col_id"),
            lit(1).alias("ind"),
        )

        # WARNING: we need to rename columns for matrix multiplication to work on pySpark
        # we can keep same name columns here because it's the same matrix being multiplied
        # however when doing Affinity*item_cooccurence we need to rename
        item_cooccurrence = (
            df_transpose.join(df, df_transpose["col_id"] == df["row_id"])
            .groupBy(df_transpose["row_id"], df["col_id"])
            .agg(sum(df_transpose["ind"] * df["ind"]).alias("value"))
            .select(
                col("row_id").alias("row_item_id"),
                col("col_id").alias("col_item_id"),
                col("value"),
            )
        )

        # filter out cooccurence counts which are below threshold
        item_cooccurrence = item_cooccurrence.filter("value>=" + str(self.threshold))

        if self.debug:
            # trigger execution
            self.time()
            cnt = item_cooccurrence.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Item cooccurrence calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        log.info("Calculating item similarity...")
        similarity_type = (
            SIM_COOCCUR if self.similarity_type is None else self.similarity_type
        )

        # compute the diagonal used later for Jaccard and Lift
        if similarity_type == SIM_LIFT or similarity_type == SIM_JACCARD:
            diagonal = item_cooccurrence.filter("row_item_id == col_item_id").select(
                col("row_item_id").alias("i"), col("value").alias("d")
            )
            diagonal.createOrReplaceTempView("diagonal")
            item_cooccurrence.createOrReplaceTempView("item_cooccurrence")

        if similarity_type == SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif similarity_type == SIM_JACCARD:
            query = (
                "select A.row_item_id, A.col_item_id, (A.value/(B.d+C.d-A.value)) as value "
                + "from item_cooccurrence as A, diagonal as B, diagonal as C "
                + "where A.row_item_id = B.i and A.col_item_id=C.i"
            )
            log.info("Running query -- " + query)
            self.item_similarity = self.spark.sql(query)
        elif similarity_type == SIM_LIFT:
            query = (
                "select A.row_item_id, A.col_item_id, (A.value/(B.d*C.d)) as value "
                + "from item_cooccurrence as A, diagonal as B, diagonal as C "
                + "where A.row_item_id = B.i and A.col_item_id=C.i"
            )
            log.info("Running query -- " + query)
            self.item_similarity = self.spark.sql(query)
        else:
            raise ValueError("Unknown similarity type: {0}".format(similarity_type))

        if self.debug and (
            similarity_type == SIM_JACCARD or similarity_type == SIM_LIFT
        ):
            # trigger execution
            self.time()
            cnt = self.item_similarity.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Item similarity calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        # Calculate raw scores with a matrix multiplication.
        log.info("Calculating recommendation scores...")
        # user_affinity * item_similarity

        # naming here matters - we need distinct row and column names across the tables
        self.scores = (
            self.affinity.join(
                self.item_similarity,
                self.affinity["col_id"] == self.item_similarity["row_item_id"],
            )
            .groupBy(self.affinity["row_id"], self.item_similarity["col_item_id"])
            .agg(
                sum(self.affinity["Affinity"] * self.item_similarity["value"]).alias(
                    "score"
                )
            )
            .select(
                col("row_id").alias("row_user_id"), col("col_item_id"), col("score")
            )
        )

        if self.debug:
            # trigger execution
            self.time()
            cnt = self.scores.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Score calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        log.info("done training")

    def recommend_k_items(self, test, top_k=10, output_pandas=False, **kwargs):
        """Recommend top K items for all users which are in the test set.

        Args:
            test: indexed test Spark dataframe
            top_k: top n items to return
            output_pandas: specify whether to convert the output dataframe to Pandas.
            **kwargs:
        """

        # first check that test set users are in the training set
        # test_users = test.select('row_id').distinct().rdd.map(lambda r: r[0]).collect()
        test.createOrReplaceTempView("df_test")
        test_users = [x[0] for x in test.select("row_id").distinct().collect()]
        # check that test users are a subset of train users based on row coordinates
        if not set(test_users) <= set(self.user_row_ids):
            msg = "SAR cannot score test set users which are not in the training set"
            logging.error(msg)
            raise ValueError(msg)

        # shorthand
        scores = self.scores

        # Mask out items in the train set.  This only makes sense for some
        # problems (where a user wouldn't interact with an item more than once).
        if self.remove_seen:
            log.info("Removing seen items...")
            # perform left outer join with smaller training set - scores are a larger dataset, scoring all users and items
            masked_scores = scores.join(
                self.df,
                (scores.row_user_id == self.df.row_id)
                & (scores.col_item_id == self.df.col_id),
                "left_outer",
            )

            # now since training set is smaller, we have nulls under its value column, i.e. item is not in the
            # training set
            masked_scores = masked_scores.where(self.col_rating + " is null").select(
                col("row_user_id"), col("col_item_id"), col("score").alias("rating")
            )

        else:
            # just rename the scores column for future reference
            masked_scores = scores.select(
                col("row_user_id"), col("col_item_id"), col("score").alias("rating")
            ).distinct()

        if self.debug:
            # trigger execution
            self.time()
            cnt = masked_scores.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Masked score calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        # select scores based on the list of row IDs
        # TODO: this is experimental - supposed to be faster than doing another join unless string list blows up
        masked_scores.createOrReplaceTempView("masked_scores")
        query = (
            "select * from masked_scores where row_user_id in ("
            + ",".join([str(x) for x in test_users])
            + ")"
        )
        masked_scores = self.spark.sql(query)

        # Get top K items and scores.
        log.info("Getting top K...")
        # TODO: try groupby row_user_id with UDF
        # row_id is the user id
        # use row_number() and now rank() to avoid situations where you get same scores for different items
        window = Window.partitionBy(masked_scores["row_user_id"]).orderBy(
            masked_scores["rating"].desc()
        )
        # WARNING: rating is an internal column name here - not passed in the user data's header
        top_scores = masked_scores.select(
            *["row_user_id", "col_item_id", "rating"],
            F.row_number().over(window).alias("top")
        ).filter(F.col("top") <= top_k)

        if self.debug:
            # trigger execution
            self.time()
            cnt = top_scores.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Top-k calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        # output a Spark dataframe
        # format somethig like [Row(UserId=463, MovieId=368226, prediction=30.296138763427734)]

        # more efficient way of doing mapping on UDFs
        user_map = create_map([lit(x) for x in chain(*self.index2user.items())])
        item_map = create_map([lit(x) for x in chain(*self.index2item.items())])

        # map back the users and items to original IDs
        top_scores = top_scores.withColumn(
            self.col_user, user_map.getItem(col("row_user_id"))
        )
        top_scores = top_scores.withColumn(
            self.col_item, item_map.getItem(col("col_item_id"))
        )

        # return top scores
        top_scores = top_scores.select(
            col(self.col_user), col(self.col_item), col("rating").alias(PREDICTION_COL)
        ).orderBy(PREDICTION_COL, ascending=False)

        if self.debug:
            # trigger execution
            self.time()
            cnt = top_scores.cache().count()
            elapsed_time = self.time()
            self.timer_log += [
                "Mapping user IDs and formatting:\t%d\trows in\t%s\tseconds -\t%f\trows per second."
                % (cnt, elapsed_time, float(cnt) / elapsed_time)
            ]

        return top_scores

    def predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set"""
        raise NotImplementedError
