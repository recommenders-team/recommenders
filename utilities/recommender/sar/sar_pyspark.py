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

from utilities.common.constants import DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, TIMESTAMP_COL
from utilities.common.constants import PREDICTION_COL

from utilities.recommender.sar import SIM_JACCARD, SIM_LIFT, SIM_COOCCUR, HASHED_USERS, HASHED_ITEMS
from utilities.recommender.sar import TIME_DECAY_COEFFICIENT, TIME_NOW, TIMEDECAY_FORMULA, THRESHOLD

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class SARpySparkReference():
    """SAR reference implementation"""

    def __init__(self, spark, remove_seen=True, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL,
                 col_rating=DEFAULT_RATING_COL, col_timestamp=TIMESTAMP_COL,
                 similarity_type=SIM_JACCARD,
                 time_decay_coefficient=TIME_DECAY_COEFFICIENT, time_now=TIME_NOW,
                 timedecay_formula=TIMEDECAY_FORMULA, threshold=THRESHOLD, debug = False):

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
        self.model_str = "sar_ref_sql"
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

        # affinity scores for the recommendation
        self.scores = None

        # indexed IDs of users which are in the training set
        self.user_row_ids = None

        # training dataframe reference
        self.df = None

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
        ap = self.spark.sql("select * from affinity where row_id = %d" % row_id).toPandas()

        ap["col_id"] = ap["col_id"] - 1
        n = ap["col_id"].max()
        affinity_vector = np.zeros((1,len(self.unique_items)))
        affinity_vector[0, ap.col_id.tolist()] = ap.Affinity.tolist()

        return affinity_vector

    def _fit(self, df):
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
        # Markus: avoid memory blow-up
        # self.user_row_ids = [x[0] for x in df.select(self.col_item).distinct().collect()]

        log.info('Collecting user affinity matrix...')

#       if self.timedecay_formula:
#           # WARNING: previously we would take the last value in training dataframe and set it
#           # as a matrix U element
#           # for each user-item pair. Now with time decay, we compute a sum over ratings given
#           # by a user in the case
#           # when T=np.inf, so user gets a cumulative sum of ratings for a particular item and
#           # not the last rating.
#           log.info('Calculating time-decayed affinities...')
#           # Time Decay
#           # do a group by on user item pairs and apply the formula for time decay there
#           # Time T parameter is in days and input time is in seconds
#           # so we do dt/60/(T*24*60)=dt/(T*24*3600)
#           # the folling is the query which we want to run
#           if self.time_now is None:
#               self.time_now = df.select(F.max(self.col_timestamp)).first()[0]
#
#           """
#           select
#           row_id, col_id, sum(Rating * exp(-log(2) * (t0 - Timestamp) / (T * 3600 * 24))) as Affinity
#           from df_train group
#           by
#           row_id, col_id
#           """
#           query = """select
#           row_id, col_id, sum(%s * exp(-log(2) * (%f - %s) / (%f * 3600 * 24))) as Affinity
#           from df_train group
#           by
#           row_id, col_id""" % (self.col_rating, self.time_now, self.col_timestamp, self.time_decay_coefficient)
#           log.info("Running query -- " + query)
#           df = self.spark.sql(query)
#
#       else:
#           # without time decay we take the last user-provided rating supplied in the dataset as the
#           # final rating for the user-item pair
#           logging.info("Deduplicating the user-item counts")
#           query = "select distinct row_id, col_id, "+self.col_rating+" as Affinity from df_train"
#           log.info("Running query -- " + query)
#           df = self.spark.sql(query)

        # record affinity scores
        self.affinity = df
        if self.debug:
            # trigger execution
            self.time()
            cnt = self.affinity.cache().count()
            elapsed_time = self.time()
            self.timer_log += ["Affinity calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second." %
                               (cnt, elapsed_time, float(cnt) / elapsed_time)]

        # create affinity
        log.info('Calculating item cooccurrence...')

        # filter out cooccurence counts which are below threshold
        query = """
        SELECT A.{col_item} i1, B.{col_item} i2, count(*) value
        FROM   df_train A INNER JOIN df_train B
               ON A.{col_user} = B.{col_user} AND A.{col_item} <= b.{col_item}  
        GROUP  BY A.{col_item}, B.{col_item}
        HAVING count(*) >= {threshold}
        """.format(col_item = self.col_item, col_user = self.col_user, threshold = self.threshold)
        
        item_cooccurrence = self.spark.sql(query).cache()
        item_cooccurrence.createOrReplaceTempView("item_cooccurrence")
 
        if self.debug:
            # trigger execution
            self.time()
            cnt = item_cooccurrence.cache().count()
            elapsed_time = self.time()
            self.timer_log += ["Item cooccurrence calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second." %
                               (cnt, elapsed_time, float(cnt)/elapsed_time)]
 
        log.info('Calculating item similarity...')
        similarity_type = (SIM_COOCCUR if self.similarity_type is None
                           else self.similarity_type)

        # compute the diagonal used later for Jaccard and Lift
        if similarity_type == SIM_LIFT or similarity_type == SIM_JACCARD:
            item_marginal = self.spark.sql("SELECT i1 i, value AS margin FROM item_cooccurrence WHERE i1 = i2")
            item_marginal.createOrReplaceTempView("item_marginal")

        if similarity_type == SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif similarity_type == SIM_JACCARD:
            query = """
            SELECT i1, i2, value / (M1.margin + M2.margin - value) AS value
            FROM item_cooccurrence A 
                INNER JOIN item_marginal M1 ON A.i1 = M1.i 
                INNER JOIN item_marginal M2 ON A.i2 = M2.i
            """
            log.info("Running query -- " + query)
            self.item_similarity = self.spark.sql(query)
        elif similarity_type == SIM_LIFT:
            query = """
            SELECT i1, i2, value / (M1.margin * M2.margin) AS value
            FROM item_cooccurrence A 
                INNER JOIN item_marginal M1 ON A.i1 = M1.i 
                INNER JOIN item_marginal M2 ON A.i2 = M2.i
            """
            log.info("Running query -- " + query)
            self.item_similarity = self.spark.sql(query)
        else:
            raise ValueError("Unknown similarity type: {0}".format(similarity_type))

        if self.debug and (similarity_type == SIM_JACCARD or similarity_type == SIM_LIFT):
            # trigger execution
            self.time()
            cnt = self.item_similarity.cache().count()
            elapsed_time = self.time()
            self.timer_log += ["Item similarity calculation:\t%d\trows in\t%s\tseconds -\t%f\trows per second." %
                               (cnt, elapsed_time, float(cnt) / elapsed_time)]

        self.item_similarity.createOrReplaceTempView("item_similarity")

        # upper-triangular to full-matrix
        query = """
        SELECT J1.i1, J1.i2, J1.value FROM item_similarity J1 
        UNION ALL 
        SELECT J2.i2 i1, J2.i1 i2, J2.value FROM item_similarity J2 WHERE J2.i1 <> J2.i2
        """
        self.item_similarity_full = self.spark.sql(query)
        self.item_similarity_full.createOrReplaceTempView("item_similarity_full")

        log.info('Calculating recommendation scores...')

        # user_affinity * item_similarity
        query = """
        SELECT df.{col_user} userID, S.i2 itemID, SUM(df.{col_rating} * S.value) AS score
        FROM   df_train df, item_similarity_full S
        WHERE  df.{col_item} = S.i1
        GROUP BY df.{col_user}, S.i2 
        """.format(col_user = self.col_user, col_item = self.col_item, col_rating = self.col_rating)
        self.scores = self.spark.sql(query)
        self.scores.createOrReplaceTempView("scores")

    def _recommend_k_items(self, test, top_k=10, output_pandas=False, **kwargs):
        """Recommend top K items for all users which are in the test set.

        Args:
            test: indexed test Spark dataframe
            top_k: top n items to return
            output_pandas: specify whether to convert the output dataframe to Pandas.
            **kwargs:
        """

        test.createOrReplaceTempView("df_test")

        # get the top items   
        query = """
        SELECT
            scores.userID, scores.itemID, scores.score
        FROM (SELECT DISTINCT {col_user} userID FROM df_test) users
            INNER JOIN scores ON users.userID = scores.userID
        """.format(col_user = self.col_user, top_k = top_k)


        # remove previously seen items
        if self.remove_seen:
            top_scores = self.spark.sql(query)        
            top_scores.createOrReplaceTempView("top_scores_full")

            query = """
            SELECT userID, itemID, score
            FROM
            (
                SELECT ts.*, df_test.{col_item} existingItemID
                FROM top_scores_full ts LEFT OUTER JOIN df_test
                    ON ts.userID = df_test.{col_user} AND ts.itemID = df_test.{col_item}
            )
            WHERE existingItemID IS NULL
            """.format(col_user = self.col_user, col_item = self.col_item)

            top_scores = self.spark.sql(query)

        # filter down to top-k items
        top_scores.createOrReplaceTempView("top_scores")

        query = """
        SELECT userID, itemID, score
        FROM
        (
            SELECT
                top_scores.*,
                row_number() OVER(PARTITION BY userID ORDER BY score DESC) rank
            FROM top_scores
        )
        WHERE rank <= {top_k} 
        """.format(col_user = self.col_user, top_k = top_k)
  
        return self.spark.sql(query)

    def _predict(self, test):
        """Output SAR scores for only the users-items pairs which are in the test set"""
        raise NotImplementedError

