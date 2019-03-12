# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from urllib.request import urlretrieve
import logging

log = logging.getLogger(__name__)

from tqdm import tqdm


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def maybe_download(url, filename, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.
    
    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        with TqdmUpTo(unit="B", unit_scale=True) as t:
            filepath, _ = urlretrieve(url, filepath, reporthook=t.update_to)
    else:
        log.debug("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def remove_filepath(filepath):
    """ Remove file. Be careful not to erase anything else. 
    
    Args:
        filepath (str): Path to the file

    """
    try:
        os.remove(filepath)
    except OSError:
        pass