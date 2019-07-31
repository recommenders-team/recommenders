# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import requests
import urllib

API_URL_WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
API_URL_WIKIDATA = "https://query.wikidata.org/sparql"

def find_wikidataID(name):
    """Find the entity ID in wikidata from a title string.

    Args:
        name (str): A string with search terms (eg. "Batman (1989) film")

    Returns:
        (str): wikidata entityID corresponding to the title string. 
                  'entityNotFound' will be returned if no page is found
    """
    url_opts = "&".join([
        "action=query",
        "list=search",
        "srsearch={}".format(urllib.parse.quote(bytes(name, encoding='utf8'))),
        "format=json",
        "prop=pageprops",
        "ppprop=wikibase_item",
    ])
    try:
        r = requests.get("{url}?{opts}".format(url=API_URL_WIKIPEDIA, opts=url_opts))
    except requests.exceptions.RequestException as err:
        print(err)
        entity_id = "entityNotFound"
        return entity_id
    try:
        pageID = r.json().get("query", {}).get("search", [{}])[0].get("pageid", "entityNotFound")
    except Exception as e:
        print("Page Name not found in Wikipedia")
        return "entityNotFound"

    if pageID == "entityNotFound":
        return "entityNotFound"

    url_opts = "&".join([
        "action=query",
        "prop=pageprops",
        "format=json",
        "pageids={}".format(pageID),
    ])
    try:
        r = requests.get("{url}?{opts}".format(url=API_URL_WIKIPEDIA, opts=url_opts))
    except requests.exceptions.RequestException as err:
        print(err)
        entity_id = "entityNotFound"
        return entity_id
    
    entity_id = r.json().get("query", {}).get("pages", {}).get(str(pageID), {}).get("pageprops", {}).get("wikibase_item", "entityNotFound")
    return entity_id

def query_entity_links(entityID):
    """Query all linked pages from a wikidata entityID

    Args:
        entityID (str): A wikidata page ID.

    Returns:
        (json): dictionary with linked pages.
    """
    query = """
    PREFIX entity: <http://www.wikidata.org/entity/>
    #partial results

    SELECT ?propUrl ?propLabel ?valUrl ?valLabel
    WHERE
    {
        hint:Query hint:optimizer 'None' .
        {	BIND(entity:"""+entityID+""" AS ?valUrl) .
            BIND("N/A" AS ?propUrl ) .
            BIND("identity"@en AS ?propLabel ) .
        }
        UNION
        {	entity:"""+entityID+""" ?propUrl ?valUrl .
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
    try:
        r = requests.get(API_URL_WIKIDATA, params = {'format': 'json', 'query': query})
    except requests.exceptions.RequestException as err:
        print(err)
        return {}
    try:
        data = r.json()
    except Exception as e:
        print(e)
        print("Entity ID not Found in Wikidata")
        return {}
    return data

def read_linked_entities(data):
    """Obtain lists of liken entities (IDs and names) from dictionary
    Args:
        data (json): dictionary with linked pages.

    Returns:
        (list): List of liked entityIDs
        (list): List of liked entity names
    """
    related_entities = []
    related_names = []
    if data == {}:
        return related_entities, related_names
    for c in data.get("results").get("bindings"):
        url = c.get("valUrl").get("value")
        related_entities.append(url.replace("http://www.wikidata.org/entity/", ""))
        name = c.get("valLabel").get("value")
        related_names.append(name)
    return related_entities, related_names

def query_entity_description(entityID):
    """Query entity wikidata description from entityID
    Args:
        entityID (str): A wikidata page ID.

    Returns:
        (str): Wikidata short description of the entityID
               descriptionNotFound' will be returned if no 
               description is found
    """
    query = """
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX schema: <http://schema.org/>

    SELECT ?o
    WHERE 
    {
      wd:"""+entityID+""" schema:description ?o.
      FILTER ( lang(?o) = "en" )
    }
    """
    try:
        r = requests.get(API_URL_WIKIDATA, params = {'format': 'json', 'query': query})
    except requests.exceptions.RequestException as err:
        print(err)
        description = "descriptionNotFound"
    
    description = r.json().get("results", {}).get("bindings", [{}])[0].get("o",{}).get("value", "descriptionNotFound")
    return description
