import requests
import urllib
import os
import pickle
import logging
import base64
from PIL import Image
from datetime import datetime, timedelta


def locate_prompt(prompt_name, module_directory):
    """
    Locates a prompt file in the prompts directory.

    Args:
        prompt_name (str): The name of the prompt file.

    Returns:
        str: The full path to the prompt file.
    """
    prompt_path = os.path.join(module_directory, 'prompts', prompt_name)
    if os.path.exists(prompt_path): return prompt_path

    prompt_path = os.path.join('prompts', prompt_name) 
    if os.path.exists(prompt_path): return prompt_path

    prompts_directory = locate_directory('prompts')
    if prompts_directory:
        prompt_path = os.path.join(prompts_directory, prompt_name)
        if os.path.exists(prompt_path): return prompt_path
        
    return os.path.join(module_directory, 'prompts', prompt_name)


def locate_directory(target_dir):
    if os.path.exists(target_dir):
        return os.path.abspath(target_dir).replace("\\", "/").replace("//", "/")

    # Helper function to search for directory upwards
    def search_upwards(start_path, target_dir):
        current_path = start_path
        while current_path != os.path.dirname(current_path):  # Continue until reaching the root directory
            if target_dir in os.listdir(current_path):
                return os.path.join(current_path, os.path.basename(target_dir))
            current_path = os.path.dirname(current_path)  # Move up one directory
        return None
    
    # Helper function to search for directory downwards
    def search_downwards(start_path, target_dir):
        for root, dirs, _ in os.walk(start_path):
            if target_dir in dirs:
                return os.path.join(root, os.path.basename(target_dir))
        return None

    # Step 0: Check if the current directory is already the target directory
    if os.path.basename(os.getcwd()) == os.path.basename(target_dir):
        print(f"Already in the target directory: {os.getcwd()}")
        return target_dir

    # Step 1: Try finding the directory by searching upwards
    start_path = os.getcwd()
    target_path = search_upwards(start_path, target_dir)
    
    # Step 2: If not found upwards, try finding it by searching downwards
    if not target_path:
        target_path = search_downwards(start_path, target_dir)

    return os.path.abspath(target_dir).replace("\\", "/").replace("//", "/")



# Function to encode an image file in base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file: 
        # Read the file and encode it in base64
        encoded_string = base64.b64encode(image_file.read())
        # Decode the base64 bytes into a string
        return encoded_string.decode('ascii')
    
    
def convert_png_to_jpg(image_path):
    if os.path.splitext(image_path)[1].lower() == '.png':
        # Open the image file
        with Image.open(image_path) as img:
            # Convert the image to RGB mode if it is in RGBA mode (transparency)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            # Define the new filename with .jpg extension
            new_image_path = os.path.splitext(image_path)[0] + '.jpg'
            # Save the image with the new filename and format
            img.save(new_image_path, 'JPEG')
            return new_image_path
    else:
        return None
    

          
def download_file(url, folder_path):
    # Extract the filename from the URL
    filename = url.split('/')[-1]

    # Create the full save path
    save_path = os.path.join(folder_path, filename)

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Make sure the directory exists
        os.makedirs(folder_path, exist_ok=True)

        # Write the content to a file
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File saved to {save_path}")
        return save_path
    else:
        print(f"Failed to retrieve the File from the url: {url}")
        return None

def is_file_or_url(s):
    # Check if the string is a URL
    parsed = urllib.parse.urlparse(s)
    is_url = bool(parsed.scheme and parsed.netloc)

    # Check if the string is a local file path
    is_file = os.path.isfile(s)

    return 'url' if is_url else 'file' if is_file else 'unknown'


def save_to_pickle(a, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_from_pickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def check_replace_extension(asset_file, new_extension):
    if os.path.exists(replace_extension(asset_file, new_extension)):
        new_file = replace_extension(asset_file, new_extension)
        return new_file
    return ""

def replace_extension(asset_path, new_extension):
    base_name = os.path.splitext(asset_path)[0].strip()
    extension = os.path.splitext(asset_path)[1].strip()

    return f"{base_name}{new_extension}"

### IMPORTANT FOR WINDOWS USERS TO SUPPORT LONG FILENAME PATHS 
### IN CASE YOU"RE USING LONG FILENAMES, AND THIS IS CAUSING AN EXCEPTION, FOLLOW THESE 2 STEPS:
# 1. change a registry setting to allow long path names on this particular Windows system (use regedit.exe): under HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem, set LongPathsEnabled to DWORD value 1
# 2. Check if the group policy setting is configured to enable long path names. Open the Group Policy Editor (gpedit.msc) and navigate to Local Computer Policy > Computer Configuration > Administrative Templates > System > Filesystem. Look for the "Enable Win32 long paths" policy and make sure it is set to "Enabled".
def write_to_file(text, text_filename, mode = 'a'):
    try:
        print(f"Writing file to full path: {os.path.abspath(text_filename)}")
        if isinstance(text_filename, str): text_filename = text_filename.replace("\\", "/")
        with open(text_filename, mode, encoding='utf-8') as file:
            file.write(text)        
    except Exception as e:
        print(f"SERIOUS ERROR: Error writing text to file: {e}")

def read_asset_file(text_filename):
    try:
        print(f"Reading file from path: {os.path.abspath(text_filename)}")
        if isinstance(text_filename, str): text_filename = text_filename.replace("\\", "/")
        with open(text_filename, 'r', encoding='utf-8') as file:
            text = file.read()
        status = True
    except Exception as e:
        text = ""
        print(f"WARNING ONLY - reading text file: {e}")
        status = False

    return text, status


def read_file(text_filename):
    try:
        print(f"Reading file from path: {os.path.abspath(text_filename)}")
        if isinstance(text_filename, str): text_filename = text_filename.replace("\\", "/")
        with open(text_filename, 'r', encoding='utf-8') as file:
            text = file.read()
        status = True
    except Exception as e:
        text = ""
        print(f"WARNING ONLY - reading text file: {e}")
        status = False

    return text


def find_certain_files(directory, extension = '.xlsx'):
    xlsx_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xlsx"):
                xlsx_files.append(os.path.join(root, file))
    return xlsx_files



from pathlib import Path

# Get the directory of the current module
module_directory = Path(__file__).parent.resolve()


def find_project_root(current_path=None, marker_files=None):
    if current_path is None:
        current_path = module_directory

    if marker_files is None:
        marker_files = ['.github', 'CONTRIBUTING.md', 'LICENSE.md', '.gitignore']

    current_path = Path(current_path).resolve()
    for parent in [current_path] + list(current_path.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    return None

def find_all_files_in_project_root(filename_pattern="*", extension_pattern="*"):
    """
    Finds all files matching the specified filename pattern and extension pattern in the project root directory and its subdirectories.

    Parameters:
    - filename_pattern: The pattern for the file name (e.g., 'config*', '*', 'data?').
    - extension_pattern: The pattern for the file extension (e.g., '.py', '.txt', '*').

    Returns:
    - A list of Path objects pointing to the matching files.
    """
    module_directory = Path(__file__).parent.resolve()
    project_root = find_project_root(module_directory)

    if project_root is None:
        print("Project root not found.")
        return []

    # Ensure extension pattern starts with '.' unless it's '*'
    if extension_pattern != '*' and not extension_pattern.startswith('.'):
        extension_pattern = '.' + extension_pattern

    # Construct the search pattern
    search_pattern = f"{filename_pattern}{extension_pattern}"

    matching_files = list(project_root.rglob(search_pattern))

    return matching_files


import uuid
import hashlib

def generate_random_uuid():
    return str(uuid.uuid4())


def generate_uuid_from_string(input_string):
    # Create a SHA-1 hash of the input string
    hash_object = hashlib.sha1(input_string.encode())
    # Use the first 16 bytes of the hash to create a UUID
    return str(uuid.UUID(bytes=hash_object.digest()[:16]))


def get_file_md5(file_name):
    with open(file_name, 'rb') as file_obj:
        file_contents = file_obj.read()
        md5 = hashlib.md5(file_contents).hexdigest()
        return str(md5)
    

import os
import shutil

def copy_file(src_file_path: str, dst_folder_path: str) -> str:
    # Ensure the source file exists
    if not os.path.isfile(src_file_path):
        raise FileNotFoundError(f"Source file not found: {src_file_path}")

    # Ensure the destination folder exists or create it if it doesn't
    os.makedirs(dst_folder_path, exist_ok=True)

    # Extract the file name from the source file path
    file_name = os.path.basename(src_file_path)

    # Construct the destination file path
    dst_file_path = os.path.join(dst_folder_path, file_name)

    # Copy the file (metadata and permissions) to the destination path
    shutil.copy2(src_file_path, dst_file_path)

    return dst_file_path


import json

def write_json_file(data: dict, file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_json_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)