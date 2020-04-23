# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azure.storage.blob import BlockBlobService
from io import StringIO
import pandas as pd


def load_csv_from_blob(blob_service, container_name, blob_name, **kwargs):
    """ Load a Pandas DataFrame from CSV in Azure Blob Storage.
    
    Args:
        blob_service (azure.storage.blob.BlockBlobService): Azure BlockBlobService for dataset.
        container_name (str): Azure storage container name.
        blob_name (str): Name of the blob located in the container.
    
    Returns:
        df (pd.DataFrame): Loaded dataframe.
    """
    # Read blob into memory
    blob = blob_service.get_blob_to_text(container_name, blob_name)

    # Load into dataframe
    df = pd.read_csv(StringIO(blob.content), **kwargs)

    return df
