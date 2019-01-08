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
        StringType,
    )
<<<<<<< HEAD
    from pyspark.sql.functions import concat_ws, col
except:
=======
except ModuleNotFoundError:
>>>>>>> staging
    pass  # so the environment without spark doesn't break


# MovieLens data have a different format for each size
class DataFormat:
    def __init__(
        self,
        sep, path, has_header=False,
        item_sep=None, item_path=None, item_has_header=False,
        # user_sep=None, user_path=None, user_has_header=False
    ):
        # Rating file
        self._sep = sep
        self._path = path
        self._has_header = has_header

        # Item file
        self._item_sep = item_sep
        self._item_path = item_path
        self._item_has_header = item_has_header

        # User file
        # self._user_sep = user_sep
        # self._user_path = user_path
        # self._user_has_header = user_has_header

    """ Rating file """
    @property
    def separator(self):
        return self._sep

    @property
    def path(self):
        return self._path

    @property
    def has_header(self):
        return self._has_header

    """ Item (Movie) file """
    @property
    def item_separator(self):
        return self._item_sep

    @property
    def item_path(self):
        return self._item_path

    @property
    def item_has_header(self):
        return self._item_has_header

    # """ User file """
    # @property
    # def user_separator(self):
    #     return self._user_sep
    #
    # @property
    # def user_path(self):
    #     return self._user_path
    #
    # @property
    # def user_has_header(self):
    #     return self._user_has_header


# 10m and 20m data do not have user data
_data_format = {
    "100k": DataFormat(
        "\t", "ml-100k/u.data", False,
        "|", "ml-100k/u.item", False,
        # "|", "ml-100k/u.user", False,
    ),
    "1m": DataFormat(
        "::", "ml-1m/ratings.dat", False,
        "::", "ml-1m/movies.dat", False,
        # "::", "ml-1m/users.dat", False,
    ),
    "10m": DataFormat(
        "::", "ml-10M100K/ratings.dat", False,
        "::", "ml-10M100K/movies.dat", False,
    ),
    "20m": DataFormat(
        ",", "ml-20m/ratings.csv", True,
        ",", "ml-20m/movies.csv", True
    ),
}

# 100K data genres index to string
_genres = (
    "unknown", "Action", "Adventure", "Animation",
    "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western"
)

# Warning and error messages
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns
    (user id, movie id, rating, and timestamp), but more than four column headers are provided.
    Will only use the first four column headers."""
WARNING_HAVE_SCHEMA_AND_HEADER = """Both schema and header are provided.
    The header argument will be ignored."""
ERROR_MOVIE_LENS_SIZE = "Invalid data size. Should be one of {100k, 1m, 10m, or 20m}"
ERROR_LOCAL_CACHE_PATH = """Local cache path only accepts a zip file path:
    use/something/like_this.zip"""
ERROR_USER_ID_TYPE = "User id should be IntegerType"
ERROR_MOVIE_ID_TYPE = "Movie id should be IntegerType"
ERROR_RATING_TYPE = "Rating should be FloatType or DoubleType"


def load_pandas_df(
    size="100k",
    header=None,
    genres_col=None,
    local_cache_path="ml.zip",
):
    """Loads the MovieLens dataset as pd.DataFrame.

    Download the dataset from http://files.grouplens.org/datasets/movielens, unzip, and load

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m")
        header (list): Dataset header. If None, use ["UserId", "MovieId", "Rating", "Timestamp"] by default.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, genres is not loaded.
        local_cache_path (str): Path where to cache the zip file locally

    Returns:
        pd.DataFrame: Dataset
    """
    if header is None or len(header) == 0:
        header = ["UserId", "MovieId", "Rating", "Timestamp"]
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    datapath, item_datapath = _load_datafile(size, local_cache_path)

    df = pd.read_csv(
        datapath,
        sep=_data_format[size].separator,
        engine="python",
        names=header,
        usecols=[*range(len(header))],
        header=0 if _data_format[size].has_header else None,
    )

    # Convert 'rating' type to float
    if len(header) > 2:
        df[header[2]] = df[header[2]].astype(float)

    # Load genres. If len(header) == 1, movie id will not be loaded and thus no need to load genres
    if genres_col is not None and len(header) > 1:
        if size == "100k":
            # 100k data's movie genres are encoded as a binary array (the last 19 fields)
            # For details, see http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
            genres_header = [*(str(i) for i in range(19))]
            item_header = [header[1]] + genres_header

            item_df = pd.read_csv(
                item_datapath,
                sep=_data_format[size].item_separator,
                engine='python',
                names=item_header,
                usecols=[0, *range(5, 24)],
                header=0 if _data_format[size].item_has_header else None,
            )

            # Convert binary-encoded genres as a '|' separated string, same to the other datasets
            item_df[genres_col] = item_df[genres_header].values.tolist()
            item_df[genres_col] = item_df[genres_col].map(
                lambda l: '|'.join([_genres[i] for i, v in enumerate(l) if v == 1])
            )

            item_df.drop(genres_header, axis=1, inplace=True)
        else:
            # [MovieId, Genres] from [MovieId, Title, Genres]
            item_df = pd.read_csv(
                item_datapath,
                sep=_data_format[size].item_separator,
                engine='python',
                names=[header[1], genres_col],
                usecols=[0, 2],
                header=0 if _data_format[size].item_has_header else None,
            )

        # Merge to rating DataFrame
        df = df.merge(item_df, on=header[1])

    return df


def load_spark_df(
    spark,
    size="100k",
    header=None,
    schema=None,
<<<<<<< HEAD
    genres_col=None,
=======
>>>>>>> staging
    local_cache_path="ml.zip",
    dbutils=None,
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
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, genres is not loaded.
        local_cache_path (str): Path where to cache the zip file locally
        dbutils (Databricks.dbutils): Databricks utility object

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
            if not isinstance(schema[2].dataType, FloatType) and not isinstance(
                schema[2].dataType, DoubleType
            ):
                raise ValueError(ERROR_RATING_TYPE)
        except IndexError:
            pass

    data_path, item_datapath = _load_datafile(size, local_cache_path)
    datapath = "file:" + data_path
    item_datapath = "file:" + item_datapath  # genres file path
    if is_databricks():
        # Move files from driver to DBFS
        _, dataname = os.path.split(_data_format[size].path)
        dbfs_datapath = "dbfs:/tmp/" + dataname
        _, item_dataname = os.path.split(_data_format[size].item_path)
        dbfs_item_datapath = "dbfs:/tmp/" + item_dataname

        try:
            dbutils.fs.mv(datapath, dbfs_datapath)
            dbutils.fs.mv(item_datapath, dbfs_item_datapath)
        except:
            raise ValueError(
                "To use on a Databricks notebook, dbutils object should be passed as an argument"
            )
        datapath = dbfs_datapath
        item_datapath = dbfs_item_datapath

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

    # Load genres. If len(header) == 1, movie id will not be loaded and thus no need to load genres
    if genres_col is not None and len(header) > 1:
        if size == "100k":
            # 100k data's movie genres are encoded as a binary array (the last 19 fields)
            # For details, see http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
            genres_header = [*(str(i) for i in range(19))]
            item_header = [header[1]] + genres_header
            item_schema = StructType([StructField(h, IntegerType()) for h in item_header])

            item_df = spark.read.csv(
                item_datapath,
                schema=item_schema,
                sep=_data_format[size].item_separator,
                header=_data_format[size].item_has_header
            )

            # Convert binary-encoded genres as a '|' separated string, same to the other datasets
            item_df = item_df.select(
                header[1],
                concat_ws('|', *(col(c) for c in genres_header)).alias(genres_col)
            )
        else:
            item_schema = StructType([
                StructField(header[1], IntegerType()),
                StructField(genres_col, StringType()),
            ])

            item_separator = _data_format[size].item_separator
            if len(item_separator) > 1:
                raw_item_data = spark.sparkContext.textFile(item_datapath)
                item_data_rdd = raw_item_data.map(lambda l: l.split(item_separator)).map(
                    lambda c: [int(c[0]), str(c[1])][:]
                )
                item_df = spark.createDataFrame(item_data_rdd, item_schema)
            else:
                item_df = spark.read.csv(
                    item_datapath,
                    schema=item_schema,
                    sep=item_separator,
                    header=_data_format[size].item_has_header
                )

        # Merge to rating DataFrame
        df = df.join(item_df, header[1], 'left')

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
    datapath = os.path.join(path, dataname)
    _, item_dataname = os.path.split(_data_format[size].item_path)
    item_datapath = os.path.join(path, item_dataname)

    with ZipFile(local_cache_path, "r") as z:
        with z.open(_data_format[size].path) as zf, open(datapath, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open(_data_format[size].item_path) as zf, open(item_datapath, 'wb') as f:
            shutil.copyfileobj(zf, f)

    _clean_up(local_cache_path)

    # Make sure a temporal data file get cleaned up when done
    atexit.register(_clean_up, datapath)
    atexit.register(_clean_up, item_datapath)

    return datapath, item_datapath


def _clean_up(filepath):
    """ Remove cached file. Be careful not to erase anything else. """
    try:
        os.remove(filepath)
    except OSError:
        pass
