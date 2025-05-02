# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import pandas as pd
import requests
import logging
from retrying import retry
import functools

logger = logging.getLogger(__name__)

API_URL_WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
API_URL_WIKIDATA = "https://query.wikidata.org/sparql"
SESSION = None


def log_retries(func):
    """Decorator that logs retry attempts. Must be applied AFTER the @retry decorator.
    
    Example usage:
        @retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
        @log_retries
        def my_function():
            # Function implementation
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not hasattr(wrapper, 'attempt'):
            wrapper.attempt = 0            
        try:
            return func(*args, **kwargs)
        except Exception as e:
            wrapper.attempt += 1
            logger.warning(f"Retrying {func.__name__}: attempt {wrapper.attempt} due to {e}")
            raise
    wrapper.attempt = 0
    return wrapper


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
@log_retries
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
        response.raise_for_status()  # Raise HTTPError for bad responses
        response_json = response.json()
        try:
            search_results = response_json["query"]["search"]
        except KeyError:
            logger.warning(f"Entity '{name}' not found (search)")
            return "entityNotFound"
        page_id = search_results[0]["pageid"]
    except Exception as e:
        logger.warning(f"REQUEST FAILED or unexpected error during search for {name}: {e}")
        raise  # Re-raise for retry

    params = dict(
        action="query",
        prop="pageprops",
        ppprop="wikibase_item",
        pageids=[page_id],
        format="json",
    )

    try:
        response = session.get(API_URL_WIKIPEDIA, params=params)
        response.raise_for_status()
        response_json = response.json()
        try:
            entity_id = response_json["query"]["pages"][str(page_id)]["pageprops"]["wikibase_item"]
        except KeyError:
            logger.warning(f"Entity '{name}' not found (pageprops)")
            return "entityNotFound"
    except Exception as e:
        logger.warning(f"REQUEST FAILED or unexpected error during pageprops fetch for {name}: {e}")
        raise  # Re-raise for retry

    return entity_id

@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
@log_retries
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
        response = session.get(API_URL_WIKIDATA, params=dict(query=query, format="json"))
        response.raise_for_status()
        data = response.json()
        # Check if bindings exist and are not empty
        if not data.get("results", {}).get("bindings"):
            logger.warning(f"ENTITY LINKS NOT FOUND: {entity_id}")
            return {}  # Return empty dict, do not retry
    except Exception as e:
        logger.warning(f"REQUEST FAILED or unexpected error querying links for {entity_id}: {e}")
        raise  # Re-raise for retry

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
@log_retries
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
        r.raise_for_status()
        bindings = r.json().get("results", {}).get("bindings", [])
        if not bindings:
            logger.warning(f"DESCRIPTION NOT FOUND for {entity_id}")
            return "descriptionNotFound"  # Return specific string, do not retry
        description = bindings[0]["o"]["value"]
        return description
    except Exception as e:
        logger.warning(f"REQUEST FAILED or unexpected error querying description for {entity_id}: {e}")
        raise  # Re-raise for retry

def search_wikidata(names, extras=None, describe=True):
    """Create DataFrame of Wikidata search results

    Args:
        names (list[str]): List of names to search for
        extras (dict(str: list)): Optional extra items to assign to results for corresponding name
        describe (bool): Optional flag to include description of entity

    Returns:
        pandas.DataFrame: Wikipedia results for all names with found entities

    """

    results = []
    for idx, name in enumerate(names):
        entity_id = find_wikidata_id(name)
        logger.info(
            "name: {name}, entity_id: {id}".format(name=name, id=entity_id)
        )

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
