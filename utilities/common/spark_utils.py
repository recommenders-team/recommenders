"""
Python Common Utils
"""
from pyspark.sql import SparkSession


def start_or_get_spark(app_name="Sample", url="local[*]", memory="10G"):
    """Start Spark if not started
    Args:
        app_name (str): sets name of the application
        url (str): url for spark master
        memory (str): size of memory for spark driver
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )

    return spark
