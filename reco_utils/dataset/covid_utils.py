# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import numpy as np
import pandas as pd
import requests


def load_pandas_df(
    azure_storage_account_name="azureopendatastorage",
    azure_storage_sas_token="",
    container_name="covid19temp",
    metadata_filename="metadata.csv",
):
    """ Loads the Azure Open Research COVID-19 dataset as a pd.DataFrame.

    The Azure COVID-19 Open Research Dataset may be found at https://azure.microsoft.com/en-us/services/open-datasets/catalog/covid-19-open-research/

    Args:
        azure_storage_account_name (str): Azure storage account name.
        azure_storage_sas_token (str): Azure storage SAS token.
        container_name (str): Azure storage container name.
        metadata_filename (str): Name of file containing top-level metadata for the dataset.
    
    Returns:
        metadata (pandas.DataFrame): Metadata dataframe.
    """

    # Load into dataframe
    uri = "https://{acct}.blob.core.windows.net/{container}/{filename}{sas}".format(
        acct=azure_storage_account_name,
        container=container_name,
        filename=metadata_filename,
        sas=azure_storage_sas_token
    )
    return pd.read_csv(uri)


def remove_duplicates(df, cols):
    """ Remove duplicated entries.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
        cols (list of str): Name of columns in which to look for duplicates.
    
    Returns:
        df (pandas.DataFrame): Pandas dataframe with duplicate rows dropped.
    
    """
    for col in cols:
        # Reset index
        df = df.reset_index(drop=True)

        # Find where the identifier variable is duplicated
        dup_rows = np.where(df.duplicated([col]) == True)[0]

        # Drop duplicated rows
        df = df.drop(dup_rows)

    return df


def remove_nan(df, cols):
    """ Remove rows with NaN values in specified column.
    
    Args:
        df (pandas.DataFrame): Pandas dataframe.
        cols (list of str): Name of columns in which to look for NaN.
    
    Returns:
        df (pandas.DataFrame): Pandas dataframe with invalid rows dropped.
    
    """
    for col in cols:
        # Convert any empty string cells to nan
        df[col].replace("", np.nan, inplace=True)

        # Remove NaN rows
        df = df[df[col].notna()]

    return df


def clean_dataframe(df):
    """ Clean up the dataframe.
    
    Args:
        df (pandas.DataFrame): Pandas dataframe.
    
    Returns:
        df (pandas.DataFrame): Cleaned pandas dataframe.
    """

    # Remove duplicated rows
    cols = ["cord_uid", "doi"]
    df = remove_duplicates(df, cols)

    # Remove rows without values in specified columns
    cols = ["cord_uid", "doi", "title", "license", "url"]
    df = remove_nan(df, cols)

    return df


def retrieve_text(
        entry, 
        container_name,
        azure_storage_account_name="azureopendatastorage",
        azure_storage_sas_token="",
):
    """ Retrieve body text from article of interest.
    
    Args:
        entry (pd.Series): A single row from the dataframe (df.iloc[n]).
        container_name (str): Azure storage container name.
        azure_storage_account_name (str): Azure storage account name.
        azure_storage_sas_token (str): Azure storage SAS token.

    Results:
        text (str): Full text of the blob as a single string.
    """

    try:
        filename = entry["pdf_json_files"] or entry["pmc_json_files"]

        # Extract text
        uri = "https://{acct}.blob.core.windows.net/{container}/{filename}{sas}".format(
            acct=azure_storage_account_name,
            container=container_name,
            filename=filename,
            sas=azure_storage_sas_token
        )

        data = requests.get(uri, headers={"Content-type": "application/json"}).json()
        text = " ".join([paragraph["text"] for paragraph in data["body_text"]])

    except:
        text = ""

    return text


def get_public_domain_text(
    df, 
    container_name,
    azure_storage_account_name="azureopendatastorage",
    azure_storage_sas_token="",
):
    """ Get all public domain text.
    
    Args:
        df (pandas.DataFrame): Metadata dataframe for public domain text.
        container_name (str): Azure storage container name.
        azure_storage_account_name (str): Azure storage account name.
        azure_storage_sas_token (str): Azure storage SAS token.

    Returns:
        df_full (pandas.DataFrame): Dataframe with select metadata and full article text.
    """
    # reset index
    df = df.reset_index(drop=True)

    # Add in full_text
    df["full_text"] = df.apply(
        lambda row: retrieve_text(
            row, 
            container_name, 
            azure_storage_account_name, 
            azure_storage_sas_token
        ), axis=1
    )

    # Remove rows with empty full_text
    empty_rows = np.where(df["full_text"] == "")[0]
    df = df.drop(empty_rows)

    # Only keep columns of interest
    df_full = df[
        [
            "cord_uid",
            "doi",
            "title",
            "publish_time",
            "authors",
            "journal",
            "url",
            "abstract",
            "full_text",
        ]
    ]
    df_full = df_full.reset_index()

    return df_full
