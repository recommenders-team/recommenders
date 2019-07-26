import sys
import json
import time
import urllib

import azure.mgmt.cosmosdb
import pydocumentdb.document_client as document_client
from azure.common.client_factory import get_client_from_cli_profile
from azureml.core import Workspace
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.webservice import Webservice, AksWebservice

from reco_utils.dataset.cosmos_cli import find_collection, read_collection, read_database, find_database


def main():
    # #################################################
    # Start User Configurations
    # #################################################
    subscription_id = sys.argv[1]
    # List of workspaces to deploy services too
    workspace_region = sys.argv[2]
    # location to store the secrets file for cosmosdb
    ws_config_path = '.'

    # #################################################
    # End User Configurations
    # #################################################

    secrets_path = ws_config_path + '/dbsecrets.json'

    dataset = "mvl"
    algo = "als"

    prefix = "ai_reco" + workspace_region
    resource_group = prefix + "_" + dataset
    workspace_name = prefix + "_" + dataset + "_aml"
    print("Resource group:", resource_group)

    # CosmosDB
    location = workspace_region
    account_name = resource_group + "-ds-sql"
    # account_name for CosmosDB cannot have "_" and needs to be less than 31 chars
    account_name = account_name.replace("_", "-")[0:min(31, len(prefix))]
    documentdb_database = "recommendations"
    documentdb_collection = "user_recommendations_" + algo

    # AzureML
    # NOTE: The name of a asset must be only letters or numerals, not contain spaces, and under 30 characters
    model_name = dataset + "-" + algo + "-reco.mml"
    service_name = dataset + "-" + algo
    # Name here must be <= 16 chars and only include letters, numbers and "-"
    aks_name = prefix.replace("_", "-")[0:min(12, len(prefix))] + '-aks'
    # add a name for the container
    container_image_name = '-'.join([dataset, algo])

    ws = Workspace.create(name=workspace_name,
                          subscription_id=subscription_id,
                          resource_group=resource_group,
                          location=workspace_region,
                          exist_ok=True)

    # persist the subscription id, resource group name, and workspace name in aml_config/config.json.
    ws.write_config(ws_config_path)

    client = get_client_from_cli_profile(azure.mgmt.cosmosdb.CosmosDB, subscription_id=subscription_id)

    async_cosmosdb_create = client.database_accounts.create_or_update(resource_group, account_name,
                                                                      {
                                                                          'location': location,
                                                                          'locations': [{
                                                                              'location_name': location
                                                                          }]
                                                                      })
    account = async_cosmosdb_create.result()

    my_keys = client.database_accounts.list_keys(
        resource_group,
        account_name
    )

    master_key = my_keys.primary_master_key
    endpoint = "https://" + account_name + ".documents.azure.com:443/"

    # db client
    client = document_client.DocumentClient(endpoint, {'masterKey': master_key})

    if not find_database(client, documentdb_database):
        db = client.CreateDatabase({'id': documentdb_database})
    else:
        db = read_database(client, documentdb_database)
    # Create collection options
    options = {
        'offerThroughput': 11000
    }

    # Create a collection
    collection_definition = {'id': documentdb_collection, 'partitionKey': {'paths': ['/id'], 'kind': 'Hash'}}
    if not find_collection(client, documentdb_database, documentdb_collection):
        collection = client.CreateCollection(db['_self'], collection_definition, options)
    else:
        collection = read_collection(client, documentdb_database, documentdb_collection)

    secrets = {
        "Endpoint": endpoint,
        "Masterkey": master_key,
        "Database": documentdb_database,
        "Collection": documentdb_collection,
        "Upsert": "true"
    }
    with open(secrets_path, "w") as file:
        json.dump(secrets, file)

    score_sparkml = """

import json
def init(local=False):
global client, collection
try:
  # Query them in SQL
  import pydocumentdb.document_client as document_client

  MASTER_KEY = '{key}'
  HOST = '{endpoint}'
  DATABASE_ID = "{database}"
  COLLECTION_ID = "{collection}"
  database_link = 'dbs/' + DATABASE_ID
  collection_link = database_link + '/colls/' + COLLECTION_ID

  client = document_client.DocumentClient(HOST, {'masterKey': MASTER_KEY})
  collection = client.ReadCollection(collection_link=collection_link)
except Exception as e:
  collection = e
def run(input_json):      

try:
  import json

  id = json.loads(json.loads(input_json)[0])['id']
  query = {'query': 'SELECT * FROM c WHERE c.id = "' + str(id) +'"' } #+ str(id)

  options = {'partitionKey':str(id)}
  document_link = 'dbs/{documentdb_database}/colls/{documentdb_collection}/docs/{0}'.format(id)
  result = client.ReadDocument(document_link, options);

except Exception as e:
    result = str(e)
return json.dumps(str(result)) #json.dumps({{"result":result}})
"""

    with open(secrets_path) as json_data:
        writeConfig = json.load(json_data)
        score_sparkml = score_sparkml.replace("{key}", writeConfig['Masterkey']).replace("{endpoint}",
                                                                                         writeConfig[
                                                                                             'Endpoint']).replace(
            "{database}", writeConfig['Database']).replace("{collection}", writeConfig['Collection']).replace(
            "{documentdb_database}", documentdb_database).replace("{documentdb_collection}", documentdb_collection)

        exec(score_sparkml)

        with open("score_sparkml.py", "w") as file:
            file.write(score_sparkml)

    mymodel = Model.register(model_path="score_sparkml.py",  # model_name,  # this points to a local file
                             model_name=model_name,
                             # this is the name the model is registered as, am using same name for both path and name.
                             description="ADB trained model",
                             workspace=ws)

    runtime = "spark-py"
    conda_file = 'myenv_sparkml.yml'
    driver_file = "score_sparkml.py"

    # image creation
    myimage_config = ContainerImage.image_configuration(execution_script=driver_file,
                                                        runtime=runtime,
                                                        conda_file=conda_file)

    image = ContainerImage.create(name=container_image_name,
                                  # this is the model object
                                  models=[mymodel],
                                  image_config=myimage_config,
                                  workspace=ws)

    # Wait for the create process to complete
    image.wait_for_creation(show_output=True)

    # Use the default configuration (can also provide parameters to customize)
    prov_config = AksCompute.provisioning_configuration()

    # Create the cluster
    aks_target = ComputeTarget.create(workspace=ws,
                                      name=aks_name,
                                      provisioning_configuration=prov_config)

    aks_target.wait_for_completion(show_output=True)

    print(aks_target.provisioning_state)
    print(aks_target.provisioning_errors)

    aks_config = AksWebservice.deploy_configuration(enable_app_insights=True)

    # Webservice creation using single command, there is a variant to use image directly as well.
    try:
        aks_service = Webservice.deploy_from_image(workspace=ws, name=service_name, deployment_config=aks_config,
                                                   image=image, deployment_target=aks_target)
        aks_service.wait_for_deployment(show_output=True)
    except Exception:
        aks_service = Webservice.list(ws)[0]

    scoring_url = aks_service.scoring_uri
    print(scoring_url)
    service_key = aks_service.get_keys()[0]

    input_data = '["{\\"id\\":\\"496\\"}"]'.encode()

    req = urllib.request.Request(scoring_url, data=input_data)
    req.add_header("Authorization", "Bearer {}".format(service_key))
    req.add_header("Content-Type", "application/json")

    tic = time.time()
    with urllib.request.urlopen(req) as result:
        res = result.readlines()
        print(res)

    toc = time.time()
    print("Full run took %.2f seconds" % (toc - tic))


if __name__ == '__main__':
    main()
