# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import warnings
import shutil
import atexit
import pandas as pd
from zipfile import ZipFile
from reco_utils.dataset.url_utils import maybe_download
from reco_utils.common.notebook_utils import is_databricks

try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        FloatType,
        DoubleType,
        LongType,
    )
except:
    pass  # so the environment without spark doesn't break


# MovieLens data have a different format for each size
class DataFormat:
    def __init__(self, sep, path, has_header=False):
        self._sep = sep
        self._path = path
        self._has_header = has_header

    @property
    def separator(self):
        return self._sep

    @property
    def path(self):
        return self._path

    @property
    def has_header(self):
        return self._has_header


_data_format = {
    "100k": DataFormat("\t", "ml-100k/u.data"),
    "1m": DataFormat("::", "ml-1m/ratings.dat"),
    "10m": DataFormat("::", "ml-10M100K/ratings.dat"),
    "20m": DataFormat(",", "ml-20m/ratings.csv", True),
}

# Warning and error messages
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns (user id, movie id, rating, and timestamp)
    but more than four column headers are provided. Will only use the first four column headers."""
WARNING_HAVE_SCHEMA_AND_HEADER = (
    "Both schema and header are provided. The header argument will be ignored."
)
ERROR_MOVIE_LENS_SIZE = "Invalid data size. Should be one of {100k, 1m, 10m, or 20m}"
ERROR_LOCAL_CACHE_PATH = (
    "Local cache path only accepts a zip file path: use/something/like_this.zip"
)
ERROR_USER_ID_TYPE = "User id should be IntegerType"
ERROR_MOVIE_ID_TYPE = "Movie id should be IntegerType"
ERROR_RATING_TYPE = "Rating should be FloatType or DoubleType"


def load_pandas_df(size="100k", header=None, local_cache_path="ml.zip"):
    """Loads the MovieLens dataset as pd.DataFrame.

    Download the dataset from http://files.grouplens.org/datasets/movielens, unzip, and load

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m")
        header (list): Dataset header. If None, use ["UserId", "MovieId", "Rating", "Timestamp"] by default.
        local_cache_path (str): Path where to cache the zip file locally

    Returns:
        pd.DataFrame: Dataset
    """
    if header is None or len(header) == 0:
        header = ["UserId", "MovieId", "Rating", "Timestamp"]
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    datapath = _load_datafile(size, local_cache_path)

    df = pd.read_csv(
        datapath,
        sep=_data_format[size].separator,
        engine='python',
        names=header,
        usecols=[*range(len(header))],
        header=0 if _data_format[size].has_header else None,
    )

    # convert 'rating' type to float
    if len(header) > 2:
        df[header[2]] = df[header[2]].astype(float)

    return df


def load_spark_df(
    spark, size="100k", header=None, schema=None, local_cache_path="ml.zip"
):
    """Loads the MovieLens dataset as pySpark.DataFrame.

    Download the dataset from http://files.grouplens.org/datasets/movielens, unzip, and load

    Args:
        spark (pySpark.SparkSession)
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m")
        header (list): Dataset header. If both schema and header is None,
            use ["UserId", "MovieId", "Rating", "Timestamp"] by default. If both header and schema are provided,
            the header argument will be ignored.
        schema (pySpark.StructType): Dataset schema. If None, use
            StructType(
                [
                    StructField("UserId", IntegerType()),
                    StructField("MovieId", IntegerType()),
                    StructField("Rating", FloatType()),
                    StructField("Timestamp", LongType()),
                ]
            )
        local_cache_path (str): Path where to cache the zip file locally

    Returns:
        pySpark.DataFrame: Dataset
    """
    if schema is None or len(schema) == 0:
        # Use header to generate schema
        if header is None or len(header) == 0:
            header = ["UserId", "MovieId", "Rating", "Timestamp"]
        elif len(header) > 4:
            warnings.warn(WARNING_MOVIE_LENS_HEADER)
            header = header[:4]

        schema = StructType()
        try:
            schema.add(StructField(header[0], IntegerType())).add(
                StructField(header[1], IntegerType())
            ).add(StructField(header[2], FloatType())).add(
                StructField(header[3], LongType())
            )
        except IndexError:
            pass
    else:
        if header is not None:
            warnings.warn(WARNING_HAVE_SCHEMA_AND_HEADER)

        if len(schema) > 4:
            warnings.warn(WARNING_MOVIE_LENS_HEADER)
            schema = schema[:4]
        try:
            # User and movie IDs should be int type
            if not isinstance(schema[0].dataType, IntegerType):
                raise ValueError(ERROR_USER_ID_TYPE)
            if not isinstance(schema[1].dataType, IntegerType):
                raise ValueError(ERROR_MOVIE_ID_TYPE)
            # Ratings should be float type
            if not isinstance(schema[2].dataType, FloatType) and not isinstance(schema[2].dataType, DoubleType):
                raise ValueError(ERROR_RATING_TYPE)
        except IndexError:
            pass

    datapath = "file:" + _load_datafile(size, local_cache_path)
    if is_databricks():
        _, dataname = os.path.split(_data_format[size].path)
        dbfs_datapath = "dbfs:/tmp/" + dataname
        dbutils.fs.mv(datapath, dbfs_datapath)
        datapath = dbfs_datapath

    # pySpark's read csv currently doesn't support multi-character delimiter, thus we manually handle that
    separator = _data_format[size].separator
    if len(separator) > 1:
        raw_data = spark.sparkContext.textFile(datapath)
        data_rdd = raw_data.map(lambda l: l.split(separator)).map(
            lambda c: [int(c[0]), int(c[1]), float(c[2]), int(c[3])][: len(schema)]
        )
        df = spark.createDataFrame(data_rdd, schema)
    else:
        df = spark.read.csv(
            datapath, schema=schema, sep=separator, header=_data_format[size].has_header
        )

    return df


def _load_datafile(size, local_cache_path):
    """ Download and extract file """

    if size not in _data_format:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)
    if not local_cache_path.endswith(".zip"):
        raise ValueError(ERROR_LOCAL_CACHE_PATH)

    path, filename = os.path.split(os.path.realpath(local_cache_path))

    # Make sure a temporal zip file get cleaned up no matter what
    atexit.register(_clean_up, local_cache_path)

    maybe_download(
        "http://files.grouplens.org/datasets/movielens/ml-" + size + ".zip",
        filename,
        work_directory=path,
    )

    _, dataname = os.path.split(_data_format[size].path)
    if dataname == "":
        # this will never happen unless someone changes _data_format
        raise ValueError("Invalid data file name.")
    datapath = os.path.join(path, dataname)

    with ZipFile(local_cache_path, "r") as z:
        with z.open(_data_format[size].path) as zf, open(datapath, 'wb') as f:
            shutil.copyfileobj(zf, f)

    _clean_up(local_cache_path)

    # Make sure a temporal data file get cleaned up when done
    atexit.register(_clean_up, datapath)

    return datapath


def _clean_up(filepath):
    """ Remove cached file. Be careful not to erase anything else. """
    try:
        os.remove(filepath)
    except OSError:
        pass
