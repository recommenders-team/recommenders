# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import os
import tarfile
from contextlib import contextmanager
from tempfile import TemporaryDirectory

try:
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
except ImportError:
    pass  # so the environment without spark doesn't break

from reco_utils.dataset.url_utils import maybe_download
from reco_utils.common.notebook_utils import is_databricks


CRITEO_URL_FULL = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
CRITEO_URL_SAMPLE = (
    "http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz"
)
CRITEO_URL = {"full": CRITEO_URL_FULL, "sample": CRITEO_URL_SAMPLE}
DEFAULT_HEADER = (
    ["label"]
    + ["int{0:02d}".format(i) for i in range(13)]
    + ["cat{0:02d}".format(i) for i in range(26)]
)


def load_pandas_df(size="sample", local_cache_path=None, header=DEFAULT_HEADER):
    """Loads the Criteo DAC dataset as pandas.DataFrame. This function download, untar, and load the dataset.

    The schema is:
    <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

    More details: http://labs.criteo.com/2013/12/download-terabyte-click-logs/

    Args:
        size (str): Dataset size. It can be "sample" or "full".
        local_cache_path (str): Path where to cache the tar.gz file locally
        header (list): Dataset header names.

    Returns:
        pd.DataFrame: Criteo DAC sample dataset.
    """
    with _real_path(local_cache_path) as path:
        filepath = download_criteo(size, path)
        filepath = extract_criteo(size, filepath)
        df = pd.read_csv(filepath, sep="\t", header=None, names=header)
    return df


def load_spark_df(
    spark,
    size="sample",
    header=DEFAULT_HEADER,
    local_cache_path=None,
    dbfs_datapath="dbfs:/FileStore/dac",
    dbutils=None,
):
    """Loads the Criteo DAC dataset as pySpark.DataFrame.

    The schema is:
    <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

    More details: http://labs.criteo.com/2013/12/download-terabyte-click-logs/

    Args:
        spark (pySpark.SparkSession): Spark session.
        size (str): Dataset size. It can be "sample" or "full".
        local_cache_path (str): Path where to cache the tar.gz file locally.
        header (list): Dataset header names.
        dbfs_datapath (str): Where to store the extracted files on Databricks.
        dbutils (Databricks.dbutils): Databricks utility object.
  
    Returns:
        pySpark.DataFrame: Criteo DAC training dataset.
    """
    with _real_path(local_cache_path) as path:
        filepath = download_criteo(size, path)
        filepath = extract_criteo(size, filepath)

        if is_databricks():
            try:
                # Driver node's file path
                tar_datapath = "file:" + filepath
                ## needs to be on dbfs to load
                dbutils.fs.cp(tar_datapath, dbfs_datapath, recurse=True)
                path = dbfs_datapath
            except:
                raise ValueError(
                    "To use on a Databricks notebook, dbutils object should be passed as an argument"
                )
        else:
            path = filepath

        schema = _get_spark_schema(header)
        df = spark.read.csv(path, schema=schema, sep="\t", header=False)
    return df


def download_criteo(size="full", work_directory="."):
    """Download criteo dataset as a compressed file.

    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample".
        work_directory (str): Working directory.

    Returns:
        str: Path of the downloaded file.

    """
    url = CRITEO_URL[size]
    return maybe_download(url, work_directory=work_directory)


def extract_criteo(size, compressed_file, path=None):
    """Extract Criteo dataset tar.

    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample".
        compressed_file (str): Path to compressed file.
        path (str): Path to extract the file.
    
    Returns:
        str: Path to the extracted file.
    
    """
    if path is None:
        folder = os.path.dirname(compressed_file)
        extracted_dir = os.path.join(folder, "dac")
    else:
        extracted_dir = path

    with tarfile.open(compressed_file) as tar:
        tar.extractall(extracted_dir)

    return _manage_data_sizes(size, extracted_dir)


def _manage_data_sizes(size, path):
    if size == "sample":
        datapath = os.path.join(path, "dac_sample.txt")
    elif size == "full":
        datapath = os.path.join(path, "train.txt")
    return datapath


def _get_spark_schema(header):
    ## create schema
    schema = StructType()
    ## do label + ints
    n_ints = 14
    for i in range(n_ints):
        schema.add(StructField(header[i], IntegerType()))
    ## do categoricals
    for i in range(26):
        schema.add(StructField(header[i + n_ints], StringType()))
    return schema


@contextmanager
def _real_path(path):
    tmp_dir = TemporaryDirectory()
    if path is None:
        path = tmp_dir.name
    else:
        path = os.path.realpath(path)

    try:
        yield path
    finally:
        tmp_dir.cleanup()
