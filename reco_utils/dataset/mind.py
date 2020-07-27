# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from reco_utils.dataset.download_utils import maybe_download, download_path, unzip_file


URL_MIND_LARGE_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip"
)
URL_MIND_LARGE_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip"
)
URL_MIND_SMALL_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
)
URL_MIND_SMALL_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
)
URL_MIND = {
    "large": (URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID),
    "small": (URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID),
}


def download_mind(size="small", dest_path=None):
    """Download MIND dataset

    Args:
        size (str): Dataset size. One of ["small", "large"]
        dest_path (str): Download path. If path is None, it will download the dataset on a temporal path
    Returns:
        str, str: Path to train and validation sets.
    """
    url_train, url_valid = URL_MIND[size]
    with download_path(dest_path) as path:
        train_path = maybe_download(url=url_train, work_directory=path)
        valid_path = maybe_download(url=url_valid, work_directory=path)
    return train_path, valid_path


def extract_mind(train_path, valid_path, train_folder="train", valid_folder="valid"):
    """Extract MIND dataset

    Args:
        train_path (str): Path to train zip file
        valid_path (str): Path to valid zip file
        train_folder (str): Destination forder for train set
        valid_folder (str): Destination forder for validation set
    """
    root_folder = os.path.basename(train_path)
    train_path = os.path.join(root_folder, train_folder)
    valid_path = os.path.join(root_folder, valid_folder)
    unzip_file(train_zip, train_path)
    unzip_file(valid_zip, valid_path)
    return train_path, valid_path

