# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# this script is called by AzureML notebooks to create workspace config file for the first run 
subscription_id = os.getenv("SUBSCRIPTION_ID", default="<my-subscription-id>")
resource_group = os.getenv("RESOURCE_GROUP", default="<my-resource-group>")
workspace_name = os.getenv("WORKSPACE_NAME", default="<my-workspace-name>")
workspace_region = os.getenv("WORKSPACE_REGION", default="eastus2")

try:
    ws = Workspace.from_config()
except:
    try:
        # access existing workspace 
        ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )
        ws.write_config()
    except:
        try:
            # create a new workspace
            ws = Workspace.create(
                name=workspace_name,
                subscription_id=subscription_id,
                resource_group=resource_group,
                location = workspace_region,
                create_resource_group = True,
                exist_ok = True
            )
            ws.write_config()
        except:
            ws = None
    

if ws is None:
    raise ValueError(
        """Failed to create AzureML workspace w/ the config info provided.
        Please check if you entered the correct id, group name and workspace name"""
    )
else:
    print("AzureML workspace name: ", ws.name)