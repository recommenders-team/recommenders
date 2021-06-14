# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pandas as pd
import requests
import logging
from retrying import retry


logger = logging.getLogger(__name__)


API_URL_WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
API_URL_WIKIDATA = "https://query.wikidata.org/sparql"
SESSION = None


def get_session(session=None):
    """Get session object

    Args:
        session (requests.Session): request session object

    Returns:
        requests.Session: request session object
    """

    if session is None:
        global SESSION
        if SESSION is None:
            SESSION = requests.Session()
        session = SESSION

    return session


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def find_wikidata_id(name, limit=1, session=None):
    """Find the entity ID in wikidata from a title string.

    Args:
        name (str): A string with search terms (eg. "Batman (1989) film")
        limit (int): Number of results to return
        session (requests.Session): requests session to reuse connections

    Returns:
        str: wikidata entityID corresponding to the title string. 'entityNotFound' will be returned if no page is found
    """

    session = get_session(session=session)

    params = dict(
        action="query",
        list="search",
        srsearch=bytes(name, encoding="utf8"),
        srlimit=limit,
        srprop="",
        format="json",
    )

    try:
        response = session.get(API_URL_WIKIPEDIA, params=params)
        page_id = response.json()["query"]["search"][0]["pageid"]
    except Exception as e:
        # TODO: distinguish between connection error and entity not found
        logger.error("ENTITY NOT FOUND")
        return "entityNotFound"

    params = dict(
        action="query",
        prop="pageprops",
        ppprop="wikibase_item",
        pageids=[page_id],
        format="json",
    )

    try:
        response = session.get(API_URL_WIKIPEDIA, params=params)
        entity_id = response.json()["query"]["pages"][str(page_id)]["pageprops"][
            "wikibase_item"
        ]
    except Exception as e:
        # TODO: distinguish between connection error and entity not found
        logger.error("ENTITY NOT FOUND")
        return "entityNotFound"

    return entity_id


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def query_entity_links(entity_id, session=None):
    """Query all linked pages from a wikidata entityID

    Args:
        entity_id (str): A wikidata entity ID
        session (requests.Session): requests session to reuse connections

    Returns:
        json: Dictionary with linked pages.
    """
    query = (
        """
    PREFIX entity: <http://www.wikidata.org/entity/>
    #partial results

    SELECT ?propUrl ?propLabel ?valUrl ?valLabel
    WHERE
    {
        hint:Query hint:optimizer 'None' .
        {	BIND(entity:"""
        + entity_id
        + """ AS ?valUrl) .
            BIND("N/A" AS ?propUrl ) .
            BIND("identity"@en AS ?propLabel ) .
        }
        UNION
        {	entity:"""
        + entity_id
        + """ ?propUrl ?valUrl .
            ?property ?ref ?propUrl .
            ?property rdf:type wikibase:Property .
            ?property rdfs:label ?propLabel
        }

        ?valUrl rdfs:label ?valLabel
        FILTER (LANG(?valLabel) = 'en') .
        OPTIONAL{ ?valUrl wdt:P18 ?picture .}
        FILTER (lang(?propLabel) = 'en' )
    }
    ORDER BY ?propUrl ?valUrl
    LIMIT 500
    """
    )

    session = get_session(session=session)

    try:
        data = session.get(
            API_URL_WIKIDATA, params=dict(query=query, format="json")
        ).json()
    except Exception as e:
        logger.error("ENTITY NOT FOUND")
        return {}

    return data


def read_linked_entities(data):
    """Obtain lists of liken entities (IDs and names) from dictionary

    Args:
        data (json): dictionary with linked pages

    Returns:
        list, list:
        - List of liked entityIDs.
        - List of liked entity names.
    """

    return [
        (
            c.get("valUrl").get("value").replace("http://www.wikidata.org/entity/", ""),
            c.get("valLabel").get("value"),
        )
        for c in data.get("results", {}).get("bindings", [])
    ]


@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def query_entity_description(entity_id, session=None):
    """Query entity wikidata description from entityID

    Args:
        entity_id (str): A wikidata page ID.
        session (requests.Session): requests session to reuse connections

    Returns:
        str: Wikidata short description of the entityID
        descriptionNotFound' will be returned if no description is found
    """
    query = (
        """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX schema: <http://schema.org/>

    SELECT ?o
    WHERE 
    {
      wd:"""
        + entity_id
        + """ schema:description ?o.
      FILTER ( lang(?o) = "en" )
    }
    """
    )

    session = get_session(session=session)

    try:
        r = session.get(API_URL_WIKIDATA, params=dict(query=query, format="json"))
        description = r.json()["results"]["bindings"][0]["o"]["value"]
    except Exception as e:
        logger.error("DESCRIPTION NOT FOUND")
        return "descriptionNotFound"

    return description


def search_wikidata(names, extras=None, describe=True, verbose=False):
    """Create DataFrame of Wikidata search results

    Args:
        names (list[str]): List of names to search for
        extras (dict(str: list)): Optional extra items to assign to results for corresponding name
        describe (bool): Optional flag to include description of entity
        verbose (bool): Optional flag to print out intermediate data

    Returns:
        pd.DataFrame: Wikipedia results for all names with found entities

    """

    results = []
    for idx, name in enumerate(names):
        entity_id = find_wikidata_id(name)
        if verbose:
            print("name: {name}, entity_id: {id}".format(name=name, id=entity_id))

        if entity_id == "entityNotFound":
            continue

        json_links = query_entity_links(entity_id)
        related_links = read_linked_entities(json_links)
        description = query_entity_description(entity_id) if describe else ""

        for related_entity, related_name in related_links:
            result = dict(
                name=name,
                original_entity=entity_id,
                linked_entities=related_entity,
                name_linked_entities=related_name,
            )
            if describe:
                result["description"] = description
            if extras is not None:
                for field, lst in extras.items():
                    result[field] = lst[idx]
            results.append(result)

    return pd.DataFrame(results)
