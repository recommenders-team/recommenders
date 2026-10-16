import os
import sys
import logging
import openai
from openai import AzureOpenAI, OpenAI
import base64
import tiktoken
import requests
import json
from typing import List
from PIL import Image
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    stop_after_delay,
    after_log
)
from rich.console import Console
console = Console()

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    stop_after_delay,
    after_log
)

from utils.openai_data_models import *
from utils.file_utils import convert_png_to_jpg, get_image_base64



def get_encoder(model = "gpt-4o"):
    if model == "gpt-45":
        return tiktoken.get_encoding("o200k_base")       
    if model == "gpt-4o":
        return tiktoken.get_encoding("o200k_base")
    if model == "gpt-4.1":
        return tiktoken.get_encoding("o200k_base")           
    if model == "o1":
        return tiktoken.get_encoding("o200k_base")       
    if model == "o1-mini":
        return tiktoken.get_encoding("o200k_base")       
    if model == "o3":
        return tiktoken.get_encoding("o200k_base")
    if model == "o3-mini":
        return tiktoken.get_encoding("o200k_base")
    if model == "o4-mini":
        return tiktoken.get_encoding("o200k_base")
    else:
        return tiktoken.get_encoding("o200k_base")


def get_token_count(text, model = "gpt-4o"):
    enc = get_encoder(model)
    return len(enc.encode(text))


def prepare_image_messages(imgs):
    img_arr = imgs if isinstance(imgs, list) else [imgs]
    img_msgs = []

    for image_path_or_url in img_arr:
        if image_path_or_url.startswith("http"):
            image = image_path_or_url
        else:
            image_path_or_url = os.path.abspath(image_path_or_url)
            try:
                if os.path.splitext(image_path_or_url)[1] == ".png":
                    image_path_or_url = convert_png_to_jpg(image_path_or_url)
                image = f"data:image/jpeg;base64,{get_image_base64(image_path_or_url)}"
            except:
                image = image_path_or_url

        img_msgs.append({ 
            "type": "image_url",
            "image_url": {
                "url": image
            }
        })
    # console.print("Image messages prepared.")
    return img_msgs


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))             
def get_embeddings(text : str, model_info: EmbeddingModelnfo = EmbeddingModelnfo()):
    if model_info.client is None: model_info = instantiate_model(model_info)
    return model_info.client.embeddings.create(input=[text], model=model_info.model_name).data[0].embedding



def call_llm(prompt: str, model_info: Union[MulitmodalProcessingModelInfo, TextProcessingModelnfo], temperature = 0.2, imgs=[]):
    content = [{"type": "text", "text": prompt}]
    content = content + prepare_image_messages(imgs)
    messages = [
        {"role": "user", "content": "You are a helpful assistant that processes text and images."},
        {"role": "user", "content": content},
    ]
    
    if model_info.client is None: model_info = instantiate_model(model_info)
    # print(">>>>>>>>>>>>>>>>> call_llm model_info", model_info)

    if (model_info.model_name == "gpt-4o") or ((model_info.model_name == "gpt-45")):
        return call_4(messages, model_info.client, model_info.model, temperature)
    elif model_info.model_name == "gpt-4.1":
        return call_41(messages, model_info.client, model_info.model, temperature)
    elif model_info.model_name == "o1":
        return call_o1(messages, model_info.client, model_info.model, model_info.reasoning_efforts)
    elif model_info.model_name == "o1-mini":
        return call_o1_mini(messages, model_info.client, model_info.model)
    elif model_info.model_name == "o3":
        return call_o3(messages, model_info.client, model_info.model, model_info.reasoning_efforts)
    elif model_info.model_name == "o3-mini":
        return call_o3_mini(messages, model_info.client, model_info.model, model_info.reasoning_efforts)
    elif model_info.model_name == "o4-mini":
        return call_o4_mini(messages, model_info.client, model_info.model, model_info.reasoning_efforts)
    else:
        return call_4(messages, model_info.client, model_info.model, temperature)


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_4(messages, client, model, temperature = 0.2):
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    result = client.chat.completions.create(model = model, temperature = temperature, messages = messages)
    return result.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_41(messages, client, model, temperature = 0.2):
    print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    result = client.chat.completions.create(model = model, temperature = temperature, messages = messages)
    return result.choices[0].message.content
      
@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_o1(messages,  client, model, reasoning_effort ="medium"): 
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.chat.completions.create(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.model_dump()['choices'][0]['message']['content']

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_o1_mini(messages,  client, model): 
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.chat.completions.create(model=model, messages=messages)
    return response.model_dump()['choices'][0]['message']['content']

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))       
def call_o3(messages,  client, model, reasoning_effort ="medium"): 
    print(f"\ncall_o3:: Calling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.chat.completions.create(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.model_dump()['choices'][0]['message']['content']

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_o3_mini(messages,  client, model, reasoning_effort ="medium"): 
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url} - Reasoning Effort: {reasoning_effort}\n")
    response = client.chat.completions.create(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.model_dump()['choices'][0]['message']['content']

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_o4_mini(messages, client, model, reasoning_effort ="medium"): 
    print(f"\ncall_o4_mini::Calling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.chat.completions.create(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.model_dump()['choices'][0]['message']['content']


def call_llm_structured_outputs(prompt: str, model_info: Union[MulitmodalProcessingModelInfo, TextProcessingModelnfo], response_format, imgs=[]):
    content = [{"type": "text", "text": prompt}]
    content = content + prepare_image_messages(imgs)
    messages = [
        {"role": "user", "content": content},
    ]

    if model_info.client is None: model_info = instantiate_model(model_info)
    # print(">>>>>>>>>>>>>>>>> call_llm_structured_outputs model_info", model_info)


    if (model_info.model_name == "gpt-4o") or ((model_info.model_name == "gpt-45")):
        return call_llm_structured_4(messages, model_info.client, model_info.model, response_format)
    elif model_info.model_name == "gpt-4.1":
        return call_llm_structured_41(messages, model_info.client, model_info.model, response_format)
    elif model_info.model_name == "o1":
        return call_llm_structured_o1(messages, model_info.client, model_info.model, response_format, model_info.reasoning_efforts)
    elif model_info.model_name == "o1-mini":
        return call_llm_structured_o1_mini(messages, model_info.client, model_info.model, response_format)
    elif model_info.model_name == "o3":
        return call_llm_structured_o3(messages, model_info.client, model_info.model, response_format, model_info.reasoning_efforts)
    elif model_info.model_name == "o3-mini":
        return call_llm_structured_o3_mini(messages, model_info.client, model_info.model, response_format, model_info.reasoning_efforts)
    elif model_info.model_name == "o4-mini":
        return call_llm_structured_o4_mini(messages, model_info.client, model_info.model, response_format, model_info.reasoning_efforts)
    else:
        return call_llm_structured_4(messages, model_info.client, model_info.model, response_format)


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_4(messages, client, model, response_format):
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    completion = client.beta.chat.completions.parse(model=model, messages=messages, response_format=response_format)
    return completion.choices[0].message.parsed

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_41(messages, client, model, response_format):
    print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    completion = client.beta.chat.completions.parse(model=model, messages=messages, response_format=response_format)
    return completion.choices[0].message.parsed

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_o1(messages, client, model, response_format, reasoning_effort ="medium"): 
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.beta.chat.completions.parse(model=model, messages=messages, reasoning_effort=reasoning_effort, response_format=response_format)
    return response.choices[0].message.parsed

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_o1_mini(messages, client, model, response_format): 
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.beta.chat.completions.parse(model=model, messages=messages, response_format=response_format)
    return response.choices[0].message.parsed

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_o3(messages, client, model, response_format, reasoning_effort ="medium"): 
    print(f"\ncall_llm_structured_o3::Calling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.beta.chat.completions.parse(model=model, messages=messages, reasoning_effort=reasoning_effort, response_format=response_format)
    return response.choices[0].message.parsed

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_o3_mini(messages, client, model, response_format, reasoning_effort ="medium"): 
    # print(f"\nCalling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.beta.chat.completions.parse(model=model, messages=messages, reasoning_effort=reasoning_effort, response_format=response_format)
    return response.choices[0].message.parsed

# @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_structured_o4_mini(messages, client, model, response_format, reasoning_effort ="medium"): 
    print(f"\ncall_llm_structured_o4_mini::Calling OpenAI APIs with {len(messages)} messages - Model: {model} - Endpoint: {client._base_url}\n")
    response = client.beta.chat.completions.parse(model=model, messages=messages, reasoning_effort=reasoning_effort, response_format=response_format)
    return response.choices[0].message.parsed


def process_function_call_result(result, functions):
    """
    Helper function to process results from function-calling completions.
    If the finish_reason is "tool_calls" or "function_call", it extracts the function call details,
    and if a mapping of functions is provided, it executes the corresponding function.
    Returns either a list of function call messages or the plain content.
    """
    finish_reason = result.choices[0].finish_reason
    if finish_reason in ["tool_calls", "function_call", "stop"]:
        # For "function_call", wrap the function_call attribute in a list.
        if (finish_reason == "function_call") or (finish_reason == "stop"):
            tool_calls = [result.choices[0].message.function_call]
        else:
            tool_calls = result.choices[0].message.tool_calls

        rets = {}
        for f in tool_calls:
            # Determine the function name and arguments based on the structure.
            if hasattr(f, "function"):
                fname = f.function.name
                fargs = f.function.arguments
            else:
                fname = f.name
                fargs = f.arguments

            if not functions:
                rets[fname] = fargs
            else:
                if fname in functions:
                    rets[fname] = functions[fname](fargs)
        
        # For building the function_call_message, extract name and arguments from the first call.
        if hasattr(tool_calls[0], "function"):
            fname = tool_calls[0].function.name
            fargs = tool_calls[0].function.arguments
        else:
            fname = tool_calls[0].name
            fargs = tool_calls[0].arguments

        call_id = getattr(tool_calls[0], "id", "function_call_default_id")
        function_call_message = {
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {
                        "name": fname,
                        "arguments": json.dumps(fargs),
                    },
                    "type": "function"
                }
            ],
            "content": "",
            "role": "assistant"
        }
        function_call_result_message = {
            "role": "tool",
            "name": fname,
            "content": json.dumps(rets),
            "tool_call_id": call_id
        }
        if not functions:
            return [function_call_message]
        else:
            return [function_call_message, function_call_result_message]
    else:
        return result.choices[0].message.content





def call_llm_functions(prompt_or_messages, tools, functions={}, temperature=0.2, model_info=None):
    """
    Top-level function to call the LLM with function calling support.
    If a plain string prompt is provided, it creates a standard system+user message.
    Depending on model_info, it dispatches to the appropriate model-specific function.
    """
    if isinstance(prompt_or_messages, str):
        prompt = prompt_or_messages
        messages = [
            {"role": "user", "content": "You are a helpful assistant, who helps the user with their query. You are designed to take a decision on which function to call. Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."},
            {"role": "user", "content": prompt}
        ]
    else:
        messages = prompt_or_messages

    if (model_info.model_name == "gpt-4o") or ((model_info.model_name == "gpt-45")):
        return call_llm_functions_4(messages, model_info, tools, functions, temperature)
    elif model_info.model_name == "gpt-4.1":
        return call_llm_functions_41(messages, model_info, tools, functions, temperature)
    elif model_info.model_name == "o1":
        return call_llm_functions_o1(messages, model_info, tools, functions)
    elif model_info.model_name == "o1-mini":
        return call_llm_functions_o1_mini(messages, model_info, tools, functions)
    elif model_info.model_name == "o3":
        return call_llm_functions_o3(messages, model_info, tools, functions)
    elif model_info.model_name == "o3-mini":
        return call_llm_functions_o3_mini(messages, model_info, tools, functions)
    elif model_info.model_name == "o4-mini":
        return call_llm_functions_o4_mini(messages, model_info, tools, functions)
    else:
        return call_llm_functions_4(messages, model_info, tools, functions, temperature)


@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_4(messages, model_info, tools, functions, temperature):
    """
    Calls the LLM (gpt-4o) with function calling enabled.
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    result = client.chat.completions.create(
        model=model_info.model,
        temperature=temperature,
        messages=messages,
        functions=tools
    )
    return process_function_call_result(result, functions)

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_41(messages, model_info, tools, functions, temperature):
    """
    Calls the LLM (gpt-4.1) with function calling enabled.
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    result = client.chat.completions.create(
        model=model_info.model,
        temperature=temperature,
        messages=messages,
        functions=tools
    )
    return process_function_call_result(result, functions)

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_o1(messages, model_info, tools, functions):
    """
    Calls the LLM (o1) with function calling enabled.
    Note: The temperature parameter is not used for o1 models.
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    response = client.chat.completions.create(
        model=model_info.model,
        messages=messages,
        functions=tools,
        reasoning_effort=model_info.reasoning_efforts
    )
    return process_function_call_result(response, functions)

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_o1_mini(messages, model_info, tools, functions):
    """
    Calls the LLM (o1-mini) with function calling enabled.
    Note: The temperature parameter is not used for o1-mini models.
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    response = client.chat.completions.create(
        model=model_info.model,
        messages=messages,
        functions=tools
    )
    return process_function_call_result(response, functions)

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_o3(messages, model_info, tools, functions):
    """
    Calls the LLM (o3) with function calling enabled.
    Note: The temperature parameter is not used for o3 models.
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    response = client.chat.completions.create(
        model=model_info.model,
        messages=messages,
        functions=tools,
        reasoning_effort=model_info.reasoning_efforts
    )
    return process_function_call_result(response, functions)

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_o3_mini(messages, model_info, tools, functions):
    """
    Calls the LLM (o3-mini) with function calling enabled.
    Note: The temperature parameter is not used for o3-mini models.
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    response = client.chat.completions.create(
        model=model_info.model,
        messages=messages,
        functions=tools,
        reasoning_effort=model_info.reasoning_efforts
    )
    return process_function_call_result(response, functions)

@retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(10))
def call_llm_functions_o4_mini(messages, model_info, tools, functions):
    """
    Calls the LLM (o4-mini) with function calling enabled.
    Note: o4-mini is a reasoning model so we include reasoning_effort param
    """
    if model_info.client is None:
        model_info = instantiate_model(model_info)
    client = model_info.client
    response = client.chat.completions.create(
        model=model_info.model,
        messages=messages,
        functions=tools,
        reasoning_effort=model_info.reasoning_efforts
    )
    return process_function_call_result(response, functions)
