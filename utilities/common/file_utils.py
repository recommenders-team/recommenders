"""
Dataset utils
"""
import os
from urllib.request import urlretrieve


def maybe_download(base_data_url, datafile, datafile_fs, verbose=False):
    """Download file if not already downloaded.
    Args:
        base_data_url (str): Base url of the file (without trailing /).
        datafile (str): File to download.
        datafile_fs (str): File path to place the downloaded file.
        verbose (bool): Verbose flag.
    """
    if os.path.isfile(datafile_fs):
        if verbose:
            print("found {} at {}".format(datafile, datafile_fs))
    else:
        if verbose:
            print("downloading {} to {}".format(datafile, datafile_fs))
        urlretrieve(base_data_url + "/" + datafile, datafile_fs)