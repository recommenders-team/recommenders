import os
import io
import atexit
import pandas as pd
from zipfile import ZipFile
from reco_utils.dataset.url_utils import maybe_download


def load_data(size="100k", header=None, local_cache_path="ml.zip"):
    """Loads the MovieLens dataset.
    Download the dataset from http://files.grouplens.org/datasets/movielens, unzip,
    and load the data as pd.DataFrame.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m")
        header (list): Dataset header. If None, use ["UserId", "MovieId", "Rating", "Timestamp"] by default.
        local_cache_path (str): Path where to cache the zip file locally

    Returns:
        pd.DataFrame: Dataset
    """
    # MovieLens data have different data-format for each size of dataset
    header_row = None
    if size == "100k":
        separator = "\t"
        datapath = "ml-100k/u.data"
    elif size == "1m":
        separator = "::"
        datapath = "ml-1m/ratings.dat"
    elif size == "10m":
        separator = "::"
        datapath = "ml-10M100K/ratings.dat"
    elif size == "20m":
        separator = ","
        header_row = 0
        datapath = "ml-20m/ratings.csv"
    else:
        raise ValueError("Invalid data size. Should be one of {100k, 1m, 10m, or 20m}")

    if header is None:
        header = ["UserId", "MovieId", "Rating", "Timestamp"]
    elif len(header) != 4:
        raise ValueError("Invalid header size. MovieLens dataset contains UserId, MovieId, Rating, and Timestamp")

    # Make sure a temporal file gets cleaned up no matter what
    atexit.register(_clean_up, local_cache_path)

    filepath = maybe_download(
        "http://files.grouplens.org/datasets/movielens/ml-" + size + ".zip", local_cache_path
    )

    with ZipFile(filepath, "r") as zf:
        with zf.open(datapath) as data:
            df = pd.read_csv(
                io.TextIOWrapper(data),
                sep=separator,
                engine='python',
                names=header,
                header=header_row,
            )
            df[header[2]] = df[header[2]].astype(float)

    _clean_up(local_cache_path)

    return df


def _clean_up(filepath):
    """ Remove cached zip file """
    try:
        os.remove(filepath)
    except OSError:
        pass
