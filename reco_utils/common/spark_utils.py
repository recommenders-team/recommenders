# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys


try:
    from pyspark.sql import SparkSession
except ImportError:
    pass  # skip this import if we are in pure python environment


def start_or_get_spark(
    app_name="Sample", 
    url="local[*]", 
    memory="10G", 
    packages=None, 
    jars=None, 
    repository=None
    ):
    """Start Spark if not started

    Args:
        app_name (str): Set name of the application
        url (str): URL for spark master
        memory (str): Size of memory for spark driver
        packages (list): list of packages to install
        jars (list): list of jar files to add
        repository (str): The maven repository

    Returns:
        obj: Spark context.
    """

    submit_args = ''
    if packages is not None:
        submit_args = '--packages {} '.format(','.join(packages))
    if jars is not None:
        submit_args += '--jars {} '.format(','.join(jars))
    if repository is not None:
        submit_args += "--repositories {}".format(repository)
    if submit_args:
        os.environ['PYSPARK_SUBMIT_ARGS'] = '{} pyspark-shell'.format(submit_args)

    spark = (
        SparkSession.builder.appName(app_name)
        .master(url)
        .config("spark.driver.memory", memory)
        .getOrCreate()
    )

    return spark
