# Databricks notebook source
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import atexit
import tarfile
import pandas as pd

from reco_utils.dataset.url_utils import maybe_download

def load_pandas_df(
        local_cache_path="dac_sample.tar.gz",
        label_col = 'Label',
        nume_cols = ['I'+str(i) for i in range(1, 14)],
        cate_cols = ['C'+str(i) for i in range(1, 27)]
):
  """Loads the Criteo DAC dataset as pandas.DataFrame.

  Download the dataset from http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz, untar, and load

  For the data, the schema is:
  
  <label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

  Args:
      local_cache_path (str): Path where to cache the tar.gz file locally
      label_col (str): The column of Label.
      nume_cols (list): The names of numerical features.
      cate_cols (list): The names of categorical features.
  Returns:
      pandas.DataFrame: Criteo DAC sample dataset.
  """

  ## download and untar the data file
  datapath = _load_datafile(local_cache_path=local_cache_path)
  df = pd.read_csv(datapath, sep="\t", header=None, names=[label_col]+nume_cols+cate_cols)
  
  return df

def _load_datafile(local_cache_path="dac_sample.tar.gz"):
    """ Download and extract file """
    path, filename = os.path.split(os.path.realpath(local_cache_path))
    # Make sure a temporal zip file get cleaned up no matter what
    atexit.register(_clean_up, local_cache_path)
    
    ## download if it doesn't already exist locally
    maybe_download(
          "http://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz",
          filename,
          work_directory=path,
    )

    #always extract to a subdirectory of cache_path called dac_sample
    # extracted_dir=os.path.join(path, "dac_sample")
    print('Extracting component files from {}.'.format(local_cache_path))
    with tarfile.open(local_cache_path) as tar:
        tar.extract('dac_sample.txt', path)
    
    _clean_up(local_cache_path)

    datapath=os.path.join(path, 'dac_sample.txt')

    atexit.register(_clean_up, datapath)
    
    return datapath

def _clean_up(filepath):
    """ Remove cached file. Be careful not to erase anything else. """
    try:
        os.remove(filepath)
    except OSError:
        pass
