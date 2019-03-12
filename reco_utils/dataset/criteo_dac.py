# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import pandas as pd
import os
import atexit
import tarfile

try:
    from pyspark.sql.types import StructType, StructField, IntegerType, StringType
except ImportError:
    pass  # so the environment without spark doesn't break

from reco_utils.dataset.url_utils import maybe_download, remove_filepath
from reco_utils.common.notebook_utils import is_databricks


CRITEO_URL_FULL = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
CRITEO_URL_SAMPLE = (
    "http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz"
)
CRITEO_URL = {"full": CRITEO_URL_FULL, "sample": CRITEO_URL_SAMPLE}
DEFAULT_HEADER = ["label"] + ["int{0:02d}".format(i) for i in range(13)] + ["cat{0:02d}".format(i) for i in range(26)]


def load_pandas_df(
    size="sample",
    local_cache_path="dac_sample.tar.gz",
    header=DEFAULT_HEADER
):
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
    path, filename = os.path.split(os.path.realpath(local_cache_path))
    download_criteo(size, filename, path)
    filepath = extract_criteo(size, path, local_cache_path)
    return pd.read_csv(filepath, sep="\t", header=None, names=header)


def load_spark_df(
    spark,
    size="sample",
    header=DEFAULT_HEADER,
    local_cache_path="dac.tar.gz",
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
    path, filename = os.path.split(os.path.realpath(local_cache_path))
    download_criteo(size, filename, path)
    filepath = extract_criteo(size, path, local_cache_path)

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
    return spark.read.csv(path, schema=schema, sep="\t", header=False)


def download_criteo(size="full", filename="dac.tar.gz", work_directory="."):
    """Download criteo dataset as a compressed file.

    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample".
        filename (str): Filename.
        work_directory (str): Working directory.

    Returns:
        str: Path of the downloaded file.

    """
    url = CRITEO_URL[size]
    return maybe_download(url, filename, work_directory)


def extract_criteo(size, path, compressed_file, remove_after_extraction=True):
    """Extract Criteo dataset tar.

    Args:
        size (str): Size of criteo dataset. It can be "full" or "sample".
        path (str): Path to the file.
        compressed_file (str): Tar file.
        remove_after_extraction (bool): Whether or not to remove the tar file after extraction.
    
    Returns:
        str: Path to the extracted file.
    """
    extracted_dir = os.path.join(path, "dac")
    print(
        "Extracting component files from {}.".format(compressed_file)
    )
    with tarfile.open(compressed_file) as tar:
        tar.extractall(extracted_dir)

    if remove_after_extraction:
        remove_filepath(compressed_file)

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




