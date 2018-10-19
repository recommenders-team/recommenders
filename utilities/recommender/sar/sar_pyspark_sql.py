"""
Reference implementation of SAR in pySpark using Spark-SQL and some dataframe operations.
This is supposed to be a super-performant implementation of SAR on Spark using pySpark.
"""

import pyspark.sql.functions as F
from pyspark.sql.types import StringType, DoubleType, StructType, StructField

SIM_COOCCUR = "cooccurrence"
SIM_JACCARD = "jaccard"
SIM_LIFT = "lift"

class SARpySparkSQLReference():
    """SAR reference implementation using SQL"""

    def __init__(self, spark, col_user='userID', col_item='itemID',
                 col_rating='rating', col_timestamp='timestamp', table_prefix=''):

        self.spark = spark
        self.header = {
            'col_user': col_user,
            'col_item': col_item, 
            'col_rating': col_rating,
            'col_timestamp': col_timestamp,
            'prefix': table_prefix
        }

    def f(self, str, **kwargs):
        return str.format(**self.header, **kwargs)

    # denominator in time decay. Zero makes time decay irrelevant
    # toggle the computation of time decay group by formula
    # current time for time decay calculation
    # cooccurrence matrix threshold
    def fit(self, df, similarity_type='jaccard',
                 time_decay_coefficient=False, time_now=None,
                 timedecay_formula=False, threshold=1,
                 ):
        """Main fit method for SAR. Expects the dataframes to have row_id, col_id columns which are indexes,
        i.e. contain the sequential integer index of the original alphanumeric user and item IDs.
        Dataframe also contains rating and timestamp as floats; timestamp is in seconds since Epoch by default.

        Arguments:
            df (pySpark.DataFrame): input dataframe which contains the index of users and items. """

        # threshold - items below this number get set to zero in coocurrence counts
        assert threshold > 0

        if timedecay_formula:
            # WARNING: previously we would take the last value in training dataframe and set it
            # as a matrix U element
            # for each user-item pair. Now with time decay, we compute a sum over ratings given
            # by a user in the case
            # when T=np.inf, so user gets a cumulative sum of ratings for a particular item and
            # not the last rating.
            # Time Decay
            # do a group by on user item pairs and apply the formula for time decay there
            # Time T parameter is in days and input time is in seconds
            # so we do dt/60/(T*24*60)=dt/(T*24*3600)
            # the folling is the query which we want to run
            df.createOrReplaceTempView("{prefix}df_train_input".format(**self.header))
    
            query = self.f("""
            SELECT
                 {col_user}, {col_item}, 
                 SUM({col_rating} * EXP(-log(2) * (latest_timestamp - CAST({col_timestamp} AS long)) / ({time_decay_coefficient} * 3600 * 24))) as {col_rating}
            FROM {prefix}df_train_input,
                 (SELECT CAST(MAX({col_timestamp}) AS long) latest_timestamp FROM {prefix}df_train_input)
            GROUP BY {col_user}, {col_item} 
            CLUSTER BY {col_user} 
             """, 
                 time_now = time_now,
                 time_decay_coefficient = time_decay_coefficient)
    
            # replace with timedecayed version
            df = self.spark.sql(query)

        df.createOrReplaceTempView(self.f("{prefix}df_train"))

        # compute co-occurrence above minimum threshold
        query = self.f("""
        SELECT A.{col_item} i1, B.{col_item} i2, COUNT(*) value
        FROM   {prefix}df_train A INNER JOIN {prefix}df_train B
               ON A.{col_user} = B.{col_user} AND A.{col_item} <= b.{col_item}  
        GROUP  BY A.{col_item}, B.{col_item}
        HAVING COUNT(*) >= {threshold}
        CLUSTER BY i1, i2
        """, threshold = threshold)
        
        item_cooccurrence = self.spark.sql(query)
        item_cooccurrence.write.mode("overwrite").saveAsTable(self.f("{prefix}item_cooccurrence"))
 
        # compute the diagonal used later for Jaccard and Lift
        if similarity_type == SIM_LIFT or similarity_type == SIM_JACCARD:
            item_marginal = self.spark.sql(self.f("SELECT i1 i, value AS margin FROM {prefix}item_cooccurrence WHERE i1 = i2"))
            item_marginal.createOrReplaceTempView(self.f("{prefix}item_marginal"))

        if similarity_type == SIM_COOCCUR:
            self.item_similarity = item_cooccurrence
        elif similarity_type == SIM_JACCARD:
            query = self.f("""
            SELECT i1, i2, value / (M1.margin + M2.margin - value) AS value
            FROM {prefix}item_cooccurrence A 
                INNER JOIN {prefix}item_marginal M1 ON A.i1 = M1.i 
                INNER JOIN {prefix}item_marginal M2 ON A.i2 = M2.i
            CLUSTER BY i1, i2
            """)
            self.item_similarity = self.spark.sql(query)
        elif similarity_type == SIM_LIFT:
            query = self.f("""
            SELECT i1, i2, value / (M1.margin * M2.margin) AS value
            FROM {prefix}item_cooccurrence A 
                INNER JOIN {prefix}item_marginal M1 ON A.i1 = M1.i 
                INNER JOIN {prefix}item_marginal M2 ON A.i2 = M2.i
            CLUSTER BY i1, i2
            """)
            self.item_similarity = self.spark.sql(query)
        else:
            raise ValueError("Unknown similarity type: {0}".format(similarity_type))

        # store upper triangular
        self.item_similarity.write.mode("overwrite").saveAsTable(self.f("{prefix}item_similarity_upper"))
        
        # expand upper triangular to full matrix
        query = self.f("""
        SELECT i1, i2, value
        FROM
        (
          (SELECT i1, i2, value FROM {prefix}item_similarity_upper)
          UNION ALL
          (SELECT i2 i1, i1 i2, value FROM {prefix}item_similarity_upper WHERE i1 <> i2)
        )
        CLUSTER BY i1
        """)
        self.item_similarity = self.spark.sql(query)
        self.item_similarity.write.mode("overwrite").saveAsTable(self.f("{prefix}item_similarity"))
        
        # free space
        self.spark.sql(self.f("DROP TABLE {prefix}item_cooccurrence"))
        self.spark.sql(self.f("DROP TABLE {prefix}item_similarity_upper"))
        
        self.item_similarity = self.spark.table(self.f("{prefix}item_similarity"))

    def get_user_affinity(self, test):
        """Prepare test set for C++ SAR prediction code.
        Find all items the test users have seen in the past.

        Arguments:
            test (pySpark.DataFrame): input dataframe which contains test users.
        """
        test.createOrReplaceTempView(self.f("{prefix}df_test"))

        query = self.f("SELECT DISTINCT {col_user} FROM {prefix}df_test CLUSTER BY {col_user}")
        
        df_test_users = self.spark.sql(query)
        df_test_users.write.mode("overwrite").saveAsTable(self.f("{prefix}df_test_users"))
        
        query = self.f("""
          SELECT a.{col_user}, a.{col_item}, CAST(a.{col_rating} AS double) {col_rating}
          FROM {prefix}df_train a INNER JOIN {prefix}df_test_users b ON a.{col_user} = b.{col_user} 
          DISTRIBUTE BY {col_user}
          SORT BY {col_user}, {col_item}          
        """)
        
        return spark.sql(query)
        
    def recommend_k_items(self, test, top_k=10, remove_seen=True):
        """Recommend top K items for all users which are in the test set.

        Args:
            test: indexed test Spark dataframe
            top_k: top n items to return
            remove_seen: remove items test users have already seen in the past from the recommended set.
        """

        # TODO: remove seen

        user_affinity = self.get_user_affinity(test)
        user_affinity.write.mode("overwrite").saveAsTable(self.f("{prefix}user_affinity"))
        
        # user_affinity * item_similarity
        # filter top-k
        query = self.f("""
        SELECT userID, itemID, score
        FROM
        (
          SELECT df.{col_user} userID,
                 S.i2 itemID,
                 SUM(df.{col_rating} * S.value) AS score,
                 row_number() OVER(PARTITION BY {col_user} ORDER BY SUM(df.{col_rating} * S.value) DESC) rank
          FROM   
            {prefix}user_affinity df, 
            {prefix}item_similarity S
          WHERE df.{col_item} = S.i1
          GROUP BY df.{col_user}, S.i2
        )
        WHERE rank <= {top_k} 
        """, top_k = top_k)

        return self.spark.sql(query)

    def write_for_sar_cpp(self, test, path):
        self.item_similarity\
            .repartition(1).sortWithinPartitions("i1", "i2")\
            .withColumn("i1", self.item_similarity["i1"].cast(StringType()))\
            .withColumn("i2", self.item_similarity["i2"].cast(StringType()))\
            .write.mode("overwrite").parquet(path + '/item_similarity.parquet')

        user_affinity = self.get_user_affinity(test)
        
        user_affinity\
            .withColumn(header['col_user'], user_affinity[header['col_user']].cast(StringType()))\
            .withColumn(header['col_item'], user_affinity[header['col_item']].cast(StringType()))\
            .withColumn(header['col_rating'], user_affinity[header['col_rating']].cast(DoubleType()))\
            .write.mode("overwrite").parquet(path + '/user-affinity.parquet')

        # TODO invoke SAR++
        # can I read from HDFS using Arrow lib?
        # can I register affinity matrix in HDFS?

        # options

        # TODO: read predictions
        # schema = StructType([
        # StructField(header['col_user'], StringType(), False),
        # StructField(header['col_item'], StringType(), False),
        # StructField('score', DoubleType(), False)
        # ])

        # pred_heavy = sqlContext.read.format('csv')\
        # .load('dbfs:/mnt/marcozo/20181016_heavy/test-prep.parquet/*.predict.csv', schema=schema, delimiter="\t")