import os
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Literal, Type, Union
from pathlib import Path
from openai import AzureOpenAI, OpenAI
from dotenv import load_dotenv
load_dotenv()


from rich.console import Console
console = Console()


def get_azure_endpoint(resource):
    print(f">>>>> https://{resource}.openai.azure.com" if not "https://" in resource else resource)
    return f"https://{resource}.openai.azure.com" if not "https://" in resource else resource


# Use unified Azure OpenAI resource, key, and API version
azure_openai_resource = os.getenv('AZURE_OPENAI_RESOURCE')
azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

azure_gpt_41_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_41', 'gpt-4.1'),
    "API_VERSION": azure_openai_api_version
}

azure_gpt_45_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_45', 'gpt-4.5-preview'),
    "API_VERSION": azure_openai_api_version
}


azure_gpt_4o_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_4O', 'gpt-4o'),
    "API_VERSION": azure_openai_api_version
}

azure_o1_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_O1', 'o1'),
    "API_VERSION": azure_openai_api_version
}


azure_o1_mini_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_O1_MINI', 'o1-mini'),
    "API_VERSION": azure_openai_api_version
}


azure_o3_mini_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_O3_MINI', 'o3-mini'),
    "API_VERSION": azure_openai_api_version
}

azure_o3_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_O3', 'o3'),
    "API_VERSION": azure_openai_api_version
}

azure_o4_mini_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_O4_MINI', 'o4-mini'),
    "API_VERSION": azure_openai_api_version
}

azure_ada_embedding_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_EMBEDDING_ADA', 'text-embedding-ada-002'),
    "API_VERSION": azure_openai_api_version,
    "DIMS": 1536
}

azure_small_embedding_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_EMBEDDING_SMALL', 'text-embedding-3-small'),
    "API_VERSION": azure_openai_api_version,
    "DIMS": 1536
}

azure_large_embedding_model_info = {
    "RESOURCE": azure_openai_resource,
    "KEY": azure_openai_key,
    "MODEL": os.getenv('AZURE_OPENAI_MODEL_EMBEDDING_LARGE', 'text-embedding-3-large'),
    "API_VERSION": azure_openai_api_version,
    "DIMS": 3072
}

openai_gpt_45_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_4O')
}

openai_gpt_4o_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_4O')
}

openai_o1_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_O1')
}

openai_o1_mini_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_O1_MINI')
}

openai_o3_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_O3', 'o3')
}

openai_o3_mini_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_O3_MINI')
}

openai_o4_mini_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_O4_MINI')
}

openai_embedding_model_info = {
    "KEY": os.getenv('OPENAI_API_KEY'),
    "MODEL": os.getenv('OPENAI_MODEL_EMBEDDING'),
    "DIMS": 3072 if os.getenv('AZURE_OPENAI_MODEL_EMBEDDING') == "text-embedding-3-large" else 1536
}



class MulitmodalProcessingModelInfo(BaseModel):
    """
    Information about the multimodal model name.
    """
    provider: Literal["azure", "openai"] = "azure"
    model_name: Literal["gpt-4o", "gpt-45", "o1", "gpt-4.1", "o3", "o4-mini"] = "gpt-4.1"
    reasoning_efforts: Optional[Literal["low", "medium", "high"]] = "medium"    
    endpoint: str = ""
    key: str = ""
    model: str = ""
    api_version: str = "2024-12-01-preview"
    client: Union[AzureOpenAI, OpenAI] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextProcessingModelnfo(BaseModel):
    """
    Information about the multimodal model name.
    """
    provider: Literal["azure", "openai"] = "azure"
    model_name: Literal["gpt-4o", "gpt-45", "o1", "o1-mini", "o3", "gpt-4.1", "o3-mini", "o4-mini"] = "gpt-4.1"
    reasoning_efforts: Optional[Literal["low", "medium", "high"]] = "medium"    
    endpoint: str = ""
    key: str = ""
    model: str = ""
    api_version: str = "2024-12-01-preview"
    client: Union[AzureOpenAI, OpenAI] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EmbeddingModelnfo(BaseModel):
    """
    Information about the multimodal model name.
    """
    provider: Literal["azure"] = "azure"
    model_name: Literal["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"] = "text-embedding-3-small"
    dimensions: Literal[1536, 3072] = 1536
    endpoint: str = ""
    key: str = ""
    model: str = ""
    api_version: str = "2024-12-01-preview"
    client: Union[AzureOpenAI, OpenAI] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)




def instantiate_model(model_info: Union[MulitmodalProcessingModelInfo, 
                                   TextProcessingModelnfo, 
                                   EmbeddingModelnfo]):
    if model_info.provider == "azure":
        if model_info.model_name == "gpt-4o":
            model_info.endpoint = get_azure_endpoint(azure_gpt_4o_model_info["RESOURCE"])
            model_info.key = azure_gpt_4o_model_info["KEY"]
            model_info.model = azure_gpt_4o_model_info["MODEL"]
            model_info.api_version = azure_gpt_4o_model_info["API_VERSION"]
        
        elif model_info.model_name == "gpt-4.1":
            model_info.endpoint = get_azure_endpoint(azure_gpt_41_model_info["RESOURCE"])
            model_info.key = azure_gpt_41_model_info["KEY"]
            model_info.model = azure_gpt_41_model_info["MODEL"]
            model_info.api_version = azure_gpt_41_model_info["API_VERSION"]
            
        elif model_info.model_name == "gpt-45":
            model_info.endpoint = get_azure_endpoint(azure_gpt_45_model_info["RESOURCE"])
            model_info.key = azure_gpt_45_model_info["KEY"]
            model_info.model = azure_gpt_45_model_info["MODEL"]
            model_info.api_version = azure_gpt_45_model_info["API_VERSION"]

        elif model_info.model_name == "o1":
            model_info.endpoint = get_azure_endpoint(azure_o1_model_info["RESOURCE"])
            model_info.key = azure_o1_model_info["KEY"]
            model_info.model = azure_o1_model_info["MODEL"]
            model_info.api_version = azure_o1_model_info["API_VERSION"]

        elif model_info.model_name == "o1-mini":
            model_info.endpoint = get_azure_endpoint(azure_o1_mini_model_info["RESOURCE"])
            model_info.key = azure_o1_mini_model_info["KEY"]
            model_info.model = azure_o1_mini_model_info["MODEL"]
            model_info.api_version = azure_o1_mini_model_info["API_VERSION"]

        elif model_info.model_name == "o3":
            model_info.endpoint = get_azure_endpoint(azure_o3_model_info["RESOURCE"])
            model_info.key = azure_o3_model_info["KEY"]
            model_info.model = azure_o3_model_info["MODEL"]
            model_info.api_version = azure_o3_model_info["API_VERSION"]

        elif model_info.model_name == "o3-mini":
            model_info.endpoint = get_azure_endpoint(azure_o3_mini_model_info["RESOURCE"])
            model_info.key = azure_o3_mini_model_info["KEY"]
            model_info.model = azure_o3_mini_model_info["MODEL"]
            model_info.api_version = azure_o3_mini_model_info["API_VERSION"]

        elif model_info.model_name == "o4-mini":
            model_info.endpoint = get_azure_endpoint(azure_o4_mini_model_info["RESOURCE"])
            model_info.key = azure_o4_mini_model_info["KEY"]
            model_info.model = azure_o4_mini_model_info["MODEL"]
            model_info.api_version = azure_o4_mini_model_info["API_VERSION"]

        elif model_info.model_name == "text-embedding-ada-002":
            model_info.endpoint = get_azure_endpoint(azure_ada_embedding_model_info["RESOURCE"])
            model_info.key = azure_ada_embedding_model_info["KEY"]
            model_info.model = azure_ada_embedding_model_info["MODEL"]
            model_info.api_version = azure_o1_mini_model_info["API_VERSION"]

        elif model_info.model_name == "text-embedding-3-small":
            model_info.endpoint = get_azure_endpoint(azure_small_embedding_model_info["RESOURCE"])
            model_info.key = azure_small_embedding_model_info["KEY"]
            model_info.model = azure_small_embedding_model_info["MODEL"]
            model_info.api_version = azure_small_embedding_model_info["API_VERSION"]

        elif model_info.model_name == "text-embedding-3-large":
            model_info.endpoint = get_azure_endpoint(azure_large_embedding_model_info["RESOURCE"])
            model_info.key = azure_large_embedding_model_info["KEY"]
            model_info.model = azure_large_embedding_model_info["MODEL"]
            model_info.api_version = azure_large_embedding_model_info["API_VERSION"]
    else:
        if model_info.model_name == "gpt-4o":
            model_info.key = openai_gpt_4o_model_info["KEY"]
            model_info.model = openai_gpt_4o_model_info["MODEL"]

        if model_info.model_name == "gpt-45":
            model_info.key = openai_gpt_45_model_info["KEY"]
            model_info.model = openai_gpt_45_model_info["MODEL"]
            
        elif model_info.model_name == "o1":
            model_info.key = openai_o1_model_info["KEY"]
            model_info.model = openai_o1_model_info["MODEL"]

        elif model_info.model_name == "o1-mini":
            model_info.key = openai_o1_mini_model_info["KEY"]
            model_info.model = openai_o1_mini_model_info["MODEL"]

        elif model_info.model_name == "o3":
            model_info.key = openai_o3_model_info["KEY"]
            model_info.model = openai_o3_model_info["MODEL"]

        elif model_info.model_name == "o3-mini":
            model_info.key = openai_o3_mini_model_info["KEY"]
            model_info.model = openai_o3_mini_model_info["MODEL"]

        elif model_info.model_name == "o4-mini":
            model_info.key = openai_o4_mini_model_info["KEY"]
            model_info.model = openai_o4_mini_model_info["MODEL"]

        elif (model_info.model_name == "text-embedding-ada-002") or \
             (model_info.model_name == "text-embedding-3-small") or \
             (model_info.model_name == "text-embedding-3-large"):
            model_info.key = openai_embedding_model_info["KEY"]
            model_info.model = openai_embedding_model_info["MODEL"]
            model_info.dimensions = openai_embedding_model_info["DIMS"]

    if model_info.provider == "azure":
        model_info.client = AzureOpenAI(azure_endpoint=model_info.endpoint, 
                                        api_key=model_info.key, 
                                        api_version=model_info.api_version)
    else:
        model_info.client = OpenAI(api_key=model_info.key)


    # console.print("Requested", model_info)
    
    return model_info
