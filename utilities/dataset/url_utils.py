import os
from urllib.request import urlretrieve


def maybe_download(filename, work_directory, url, expected_bytes=None):
    """Download a file if it is not already downloaded.
    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.
    Returns:
        filepath (str): File path of the file downloaded.
    """
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(url, filepath)
    else:
        if verbose:
            print("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            raise Exception("Failed to verify {}".format(filepath))

    return filepath
