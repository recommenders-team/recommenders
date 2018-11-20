# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import pydocumentdb.errors as errors


def find_collection(client, dbid, id):
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

