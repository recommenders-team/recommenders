# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from azure.storage.blob import BlockBlobService
import pandas as pd
import numpy as np
import json


def get_blob_service(azure_storage_account_name, azure_storage_sas_token, container_name):
    """ Get the Azure blob service for accessing the dataset.
    
    Args:
        azure_storage_account_name (str): Azure storage account name.
        azure_storage_sas_token (str): Azure storage SaS token.
        container_name (str): Azure storage container name.
    
    Returns:
        blob_service (azure.storage.blob.BlockBlobService): Azure BlockBlobService for dataset.
    """

    # create a blob service
    blob_service = BlockBlobService(
        account_name=azure_storage_account_name,
        sas_token=azure_storage_sas_token,
    )
    
    return blob_service

def download_metadata(blob_service, container_name, metadata_filename):
    """ Download the metadata file.
    
    Args:
        blob_service (azure.storage.blob.BlockBlobService): Azure BlockBlobService for dataset.
        container_name (str): Azure storage container name.
        metadata_filename (str): Name of the metadata file located in the remote dataset.
    
    Returns:
        metadata (pd.DataFrame): Metadata dataframe.
    """
    
    # download metadata.csv
    blob_service.get_blob_to_path(
        container_name=container_name,
        blob_name=metadata_filename,
        file_path=metadata_filename
    )

    # read into a dataframe
    metadata = pd.read_csv(metadata_filename)
    
    return metadata

def extract_public_domain(metadata):
    """ Only keep rows containing public domain articles.
    
    Args:
        metadata (pd.DataFrame): Metadata dataframe.
    
    Returns:
        metadata_public (pd.DataFrame): Dataframe only containing rows of public domain articles.
    """
    metadata_public = metadata.loc[metadata['license']=='cc0']
    
    return metadata_public

def remove_duplicates(df, col='cord_uid'):
    """ Remove duplicated entries.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
        col (str): Name of column in which to look for duplicates.
    
    Returns:
        df (pd.DataFrame): Pandas dataframe with duplicate rows dropped.
    
    """
    # Find where the identifier variable is duplicated
    dup_rows = np.where(df.duplicated([col])==True)[0]
    
    # Drop duplicated rows
    df = df.drop(dup_rows)
    
    return df

def remove_nan(df, col='url'):
    """ Remove rows with NaN values in specified column.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
        col (str): Name of column in which to look for NaN.
    
    Returns:
        df (pd.DataFrame): Pandas dataframe with invalid rows dropped.
    
    """
    
    # Remove NaN rows
    df = df[df[col].notna()]
    
    return df

def clean_dataframe(df):
    """ Clean up the dataframe.
    
    Args:
        df (pd.DataFrame): Pandas dataframe.
    
    Returns:
        df (pd.DataFrame): Cleaned pandas dataframe.
    """
    
    # Remove duplicated rows
    df = remove_duplicates(df)
    
    # Remove rows without URLs
    df = remove_nan(df)
    
    return df

def extract_text_from_file(blob_service, container_name, blob_name):
    """ Extract the body text from the blob.
    
    Args:
        blob_service (azure.storage.blob.BlockBlobService): Azure BlockBlobService for dataset.
        container_name (str): Name of Azure storage container.
        blob_name (str): Name of blob to access.
    
    Results:
        full_text_flat (str): Full body text as a single string.
    """
    
    blob_as_json_string = blob_service.get_blob_to_text(container_name=container_name, blob_name=blob_name)
    data = json.loads(blob_as_json_string.content)
    
    # the text itself lives under 'body_text'
    full_text = data['body_text']

    # many NLP tasks play nicely with a list of sentences
    sentences = list()
    for paragraph in full_text:
        sentences.append(paragraph['text'])
        
    full_text_flat = ''
    for sentence in sentences:
        full_text_flat = full_text_flat + ' ' + sentence
    
    return full_text_flat

def retrieve_text(entry, blob_service, container_name):
    """ Retrieve body text from article of interest.
    
    Args:
        entry (pd.Series): A single row from the dataframe (df.iloc[n]).
        blob_service (azure.storage.blob.BlockBlobService): Azure BlockBlobService for dataset.
        container_name (str): Azure storage container name.
    
    Results:
        text (str): Full text of the blob as a single string.
    """
    
    try:
        # select based on whether it's pdf or pmc_xml
        if entry['has_pdf_parse'] == True:
            blob_name = '{0}/pdf_json/{1}.json'.format(entry['full_text_file'], entry['sha'])
        else:
            if entry['has_pmc_xml_parse'] == True:
                blob_name = '{0}/pmc_json/{1}.xml.json'.format(entry['full_text_file'], entry['pmcid']) 
            else:
                print('Neither PDF or PMC_XML data is available for this file')

        # Extract text
        text = extract_text_from_file(blob_service, container_name, blob_name)
    except:
        text = ''
    
    return text

def get_public_domain_text(df, blob_service, container_name):
    """ Get all public domain text.
    
    Args:
        df (pd.DataFrame): Metadata dataframe for public domain text.
        blob_service (azure.storage.blob.BlockBlobService): Azure BlockBlobService for dataset.
        container_name (str): Azure storage container name.
    
    Returns:
        df_full (pd.DataFrame): Dataframe with select metadata and full article text.
    """
    # reset index
    df = df.reset_index()

    # Add new column to fill
    df['full_text'] = np.nan

    # Add in full_text
    for row in range(0, len(df)):
        df['full_text'][row] = retrieve_text(df.iloc[row], blob_service, container_name)
        
    # Remove rows with empty full_text
    empty_rows = np.where(df['full_text']=='')[0]
    df = df.drop(empty_rows)
    
    # Only keep columns of interest
    df_full = df[['cord_uid','doi','title','publish_time','authors','journal','url','abstract','full_text']]
    df_full = df_full.reset_index()
    
    return df_full