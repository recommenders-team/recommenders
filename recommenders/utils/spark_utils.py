# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


try:
    from pyspark.sql import SparkSession  # noqa: F401
except ImportError:
    pass  # skip this import if we are in pure python environment

MMLSPARK_PACKAGE = "com.microsoft.azure:synapseml_2.12:0.9.4"
MMLSPARK_REPO = "https://mmlspark.azureedge.net/maven"
# We support Spark v3, but in case you wish to use v2, set
# MMLSPARK_PACKAGE = "com.microsoft.ml.spark:mmlspark_2.11:0.18.1"
# MMLSPARK_REPO = "https://mvnrepository.com/artifact"


def start_or_get_spark(
    app_name="Sample",
    url="local[*]",
    memory="10g",
    config=None,
    packages=None,
    jars=None,
    repositories=None,
):
    """Start Spark if not started

    Args:
        app_name (str): set name of the application
        url (str): URL for spark master
        memory (str): size of memory for spark driver
        config (dict): dictionary of configuration options
        packages (list): list of packages to install
        jars (list): list of jar files to add
        repositories (list): list of maven repositories

    Returns:
        object: Spark context.
    """

    submit_args = ""
    if packages is not None:
        submit_args = "--packages {} ".format(",".join(packages))
    if jars is not None:
        submit_args += "--jars {} ".format(",".join(jars))
    if repositories is not None:
        submit_args += "--repositories {}".format(",".join(repositories))
    if submit_args:
        os.environ["PYSPARK_SUBMIT_ARGS"] = "{} pyspark-shell".format(submit_args)

    spark_opts = [
        'SparkSession.builder.appName("{}")'.format(app_name),
        'master("{}")'.format(url),
    ]

    if config is not None:
        for key, raw_value in config.items():
            value = (
                '"{}"'.format(raw_value) if isinstance(raw_value, str) else raw_value
            )
            spark_opts.append('config("{key}", {value})'.format(key=key, value=value))

    if config is None or "spark.driver.memory" not in config:
        spark_opts.append('config("spark.driver.memory", "{}")'.format(memory))

    spark_opts.append("getOrCreate()")
    return eval(".".join(spark_opts))
