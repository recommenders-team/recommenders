# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pyspark.sql import SparkSession


def start_or_get_spark(app_name="Sample", url="local[*]", memory="10G"):
    """Start Spark if not started

    Args:
        app_name (str): Set name of the application
        url (str): URL for spark master.
        memory (str): Size of memory for spark driver.
        
    Returns:
        obj: Spark context.
    """
    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )

    return spark
