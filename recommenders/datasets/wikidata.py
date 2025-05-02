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


def retry_with_logging(**kwargs):
    """Custom retry decorator that logs retry attempts
    
    Args:
        **kwargs: Arguments to pass to the retry decorator
        
    Returns:
        Decorator function that adds logging to retries
    """
    retry_decorator = retry(**kwargs)
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = [0]
            
            def on_retry(retry_state):
                attempt[0] += 1
                logger.warning(
                    f"Retrying {func.__name__}: attempt {attempt[0]} after {retry_state.retry_object.statistics['delay_since_first_attempt_ms']/1000:.2f}s"
                )
                return True
                
            # Apply the retry decorator with our custom on_retry callback
            wrapped = retry(
                wait_func=kwargs.get('wait_func', None),
                wait_random_min=kwargs.get('wait_random_min', 1000),
                wait_random_max=kwargs.get('wait_random_max', 5000),
                stop_max_attempt_number=kwargs.get('stop_max_attempt_number', 5),
                retry_on_result=kwargs.get('retry_on_result', None),
                retry_on_exception=kwargs.get('retry_on_exception', None),
                wrap_exception=kwargs.get('wrap_exception', False),
                stop_func=kwargs.get('stop_func', None),
                wait_jitter_max=kwargs.get('wait_jitter_max', None),
                on_retry=on_retry
            )(func)
            
            return wrapped(*args, **kwargs)
        return wrapper
    return decorator


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


@retry_with_logging(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
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
    except Exception:
        # TODO: distinguish between connection error and entity not found
        logger.warning("ENTITY NOT FOUND")
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
    except Exception:
        # TODO: distinguish between connection error and entity not found
        logger.warning("ENTITY NOT FOUND")
        return "entityNotFound"

    return entity_id


@retry_with_logging(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
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
    except Exception as e:  # noqa: F841
        logger.warning("ENTITY NOT FOUND")
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


@retry_with_logging(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
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
    except Exception as e:  # noqa: F841
        logger.warning("DESCRIPTION NOT FOUND")
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
        pandas.DataFrame: Wikipedia results for all names with found entities

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
