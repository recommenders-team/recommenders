# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import ast
import os


try:
    from pyspark.sql import SparkSession
except ImportError:
    pass  # skip this import if we are in pure python environment


MMLSPARK_PACKAGE = "com.microsoft.ml.spark:mmlspark_2.11:0.18.1"
MMLSPARK_REPO = "https://mvnrepository.com/artifact"

def start_or_get_spark(
    app_name="Sample",
    url="local[*]",
    memory="10g",
    config=None,
    packages=None,
    jars=None,
    repositories=None,
    mmlspark_package=None,
    mmlspark_repository=None
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
        mmlspark_package (str): mmlspark package coordinates, if applicable
        mmlspark_repository (str): mmlspark repository, if applicable   

    Returns:
        object: Spark context.
    """

    submit_args = ""
    new_packages = packages.copy() if isinstance(packages, list) else []
    new_repos = repositories.copy() if isinstance(repositories, list) else []
    if mmlspark_package is not None:
        new_packages.append(mmlspark_package)
    if mmlspark_repository is not None:
        new_repos.append(mmlspark_repository)
    if len(new_packages):
        submit_args = "--packages {} ".format(",".join(new_packages))
    if jars is not None:
        submit_args += "--jars {} ".format(",".join(jars))
    if len(new_repos):
        submit_args += "--repositories {}".format(",".join(new_repos))
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
