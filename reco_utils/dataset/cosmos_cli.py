# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pydocumentdb.errors as errors


def find_collection(client, dbid, id):
    """Find whether or not a CosmosDB collection exists.
    
    Args:
        client (object): A pydocumentdb client object.
        dbid (str): Database ID.
        id (str): Collection ID.
    
    Returns:
        bool: True if the collection exists, False otherwise.
    """
    database_link = "dbs/" + dbid
    collections = list(
        client.QueryCollections(
            database_link,
            {
                "query": "SELECT * FROM r WHERE r.id=@id",
                "parameters": [{"name": "@id", "value": id}],
            },
        )
    )
    if len(collections) > 0:
        return True
    else:
        return False


def read_collection(client, dbid, id):
    """Read a CosmosDB collection.
    
    Args:
        client (object): A pydocumentdb client object.
        dbid (str): Database ID.
        id (str): Collection ID.
    
    Returns:
        object: A collection.
    """
    try:
        database_link = "dbs/" + dbid
        collection_link = database_link + "/colls/{0}".format(id)
        collection = client.ReadCollection(collection_link)
        return collection
    except errors.DocumentDBError as e:
        if e.status_code == 404:
            print("A collection with id '{0}' does not exist".format(id))
        else:
            raise errors.HTTPFailure(e.status_code)


def read_database(client, id):
    """Read a CosmosDB database.
    
    Args:
        client (object): A pydocumentdb client object.
        id (str): Database ID.
    
    Returns:
        object: A database.
    """
    try:
        database_link = "dbs/" + id
        database = client.ReadDatabase(database_link)
        return database
    except errors.DocumentDBError as e:
        if e.status_code == 404:
            print("A database with id '{0}' does not exist".format(id))
        else:
            raise errors.HTTPFailure(e.status_code)


def find_database(client, id):
    """Find whether or not a CosmosDB database exists.
    
    Args:
        client (object): A pydocumentdb client object.
        id (str): Database ID.
    
    Returns:
        bool: True if the database exists, False otherwise.
    """
    databases = list(
        client.QueryDatabases(
            {
                "query": "SELECT * FROM r WHERE r.id=@id",
                "parameters": [{"name": "@id", "value": id}],
            }
        )
    )
    if len(databases) > 0:
        return True
    else:
        return False
