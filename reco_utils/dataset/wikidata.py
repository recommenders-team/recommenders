# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import requests

def find_wikidataID(name):
    """Find the entity ID in wikidata from a title string.

    Args:
        name (str): A string with search terms (eg. "Batman (1989) film")
    Returns:
        (str): wikidata entityID corresponding to the title string. 
                  'entityNotFound' will be returned if no page is found
    """
    r = requests.get("https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch="+name+"&format=json&prop=pageprops&ppprop=wikibase_item")
    try:
        pageID = r.json()["query"]["search"][0]["pageid"]
        r = requests.get("https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&format=json&pageids="+str(pageID))
        try:
            entity_id = r.json()["query"]["pages"][str(pageID)]["pageprops"]["wikibase_item"]
        except:
            entity_id = "entityNotFound"
    except:
            entity_id = "entityNotFound"
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
    url = 'https://query.wikidata.org/sparql'
    r = requests.get(url, params = {'format': 'json', 'query': query})
    data = r.json()
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
    for c in data["results"]["bindings"]:
        related_entities.append(c["valUrl"]["value"].replace("http://www.wikidata.org/entity/", ""))
        related_names.append(c["valLabel"]["value"])
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
    url = 'https://query.wikidata.org/sparql'
    r = requests.get(url, params = {'format': 'json', 'query': query})
    try:
        description = r.json()["results"]["bindings"][0]["o"]["value"]
    except:
        description = "descriptionNotFound"
    return description