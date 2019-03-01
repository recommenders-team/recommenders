# Databricks notebook source
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import atexit
import tarfile

from reco_utils.dataset.url_utils import maybe_download
from reco_utils.common.notebook_utils import is_databricks
from reco_utils.common.python_utils import _clean_up

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        StringType,
    )
except ImportError:
    pass  # so the environment without spark doesn't break

def load_spark_df(
    spark,
    type='train',
    local_cache_path="dac.tar.gz",
    dbfs_datapath="dbfs:/FileStore/dac", 
    dbutils=None,
):
  """Loads the Criteo DAC dataset as pySpark.DataFrame.

  Download the dataset from https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz, untar, and load

  Args:
      spark (pySpark.SparkSession)
      type (str): which dataset to load. 2 options: 'train' or 'test'
      local_cache_path (str): Path where to cache the tar.gz file locally (not dbfs)
      dbfs_datapath: where to store the extracted files
      dbutils (Databricks.dbutils): Databricks utility object
  Returns:
      pySpark.DataFrame: Criteo DAC training dataset.
  """

  ## Exit if it isn't on databricks right now.
  if not is_databricks():
      raise ValueError("This is only supported on Databricks at the moment.")

  ## download and untar the train and test files
  extracted_tar_dir_path = _load_datafile(local_cache_path=local_cache_path, dbutils=dbutils)
  # Driver node's file path
  tar_datapath = "file:" + extracted_tar_dir_path

  try:
      ## needs to be on dbfs to load
      dbutils.fs.cp(tar_datapath, dbfs_datapath, recurse = True)
  except:
      raise ValueError("To use on a Databricks notebook, dbutils object should be passed as an argument")

  if type is 'train':
      include_label = True
      datapath = os.path.join(dbfs_datapath,'train.txt')
  elif type is 'test':
      include_label = False
      datapath = os.path.join(dbfs_datapath,'test.txt')
  else:
      raise ValueError('Unknown type. Only "train" or "test" is allowed.')

  schema = _get_schema(include_label)

  df = spark.read.csv(
       datapath, schema=schema, sep="\t", header=False
  )

  return df


def _get_schema(include_label=True):
  """ 
  Create the schema.
  
  For the training data, the schema is:
  
  <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>
  
  For the test data, the schema is missing <label>
  
  Run as _get_schema() to create the schema for the training data.
  Run as _get_schema(False) to create the schema for the testing data.
  """
  ## construct the variable names
  if include_label:
    header = ['label']
  else:
    header = []
  header.extend(['int{0:02d}'.format(i) for i in range(13)])
  header.extend(['cat{0:02d}'.format(i) for i in range(26)])
  ## create schema
  schema = StructType()
  ## do label + ints
  n_ints = 14 - (not include_label)
  for i in range(n_ints):
    schema.add(
      StructField(header[i], IntegerType())
    )
  ## do categoricals
  for i in range(26):
    schema.add(
      StructField(header[i+n_ints], StringType())
    )
  return schema


def _load_datafile(local_cache_path="dac.tar.gz", dbutils=None):
    """ Download and extract file """

    path, filename = os.path.split(os.path.realpath(local_cache_path))

    # Make sure the temporal tar file gets cleaned up no matter what
    atexit.register(_clean_up, local_cache_path)

    ## download if it doesn't already exist locally
    maybe_download(
          "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz",
          filename,
          work_directory=path,
    )

    #always extract to a subdirectory of cache_path called dac
    extracted_dir=os.path.join(path, "dac")
    print('Extracting component files from {}. This can take 3-5 minutes.'.format(local_cache_path))
    with tarfile.open(local_cache_path) as tar:
        tar.extractall(extracted_dir)
    train_file = os.path.join(extracted_dir,'train.txt')
    test_file = os.path.join(extracted_dir,'test.txt')
    
    _clean_up(local_cache_path)

    # Make sure a temporal data file get cleaned up when done
    atexit.register(_clean_up, train_file)
    atexit.register(_clean_up, test_file)

    return extracted_dir

