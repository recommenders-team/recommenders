import subprocess
import os
import sys
import glob
import json
import numpy as np
import time
from ast import literal_eval


def get_number_processors():
    """Get the number of processors in a CPU.
    Returns:
        num (int): Number of processors.
    Examples:
        >>> get_number_processors()
        4
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing #force exception in case mutiprocessing is not installed
        num = multiprocessing.cpu_count()
    return num


def get_gpu_name():
    """Get the GPUs in the system
    Examples:
        >>> get_gpu_name()
        ['Tesla M60', 'Tesla M60', 'Tesla M60', 'Tesla M60']
        
    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)


def get_gpu_memory():
    """Get the memory of the GPUs in the system
    Examples:
        >>> get_gpu_memory()
        ['8123 MiB', '8123 MiB', '8123 MiB', '8123 MiB']

    """
    try:
        out_str = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv"], stdout=subprocess.PIPE).stdout
        out_list = out_str.decode("utf-8").replace('\r','').split('\n')
        out_list = out_list[1:-1]
        return out_list
    except Exception as e:
        print(e)

        
def get_cuda_version():
    """Get the CUDA version
    Examples:
        >>> get_cuda_version()
        'CUDA Version 8.0.61'

    """
    if sys.platform == 'win32':
        raise NotImplementedError("Implement this!")
    elif sys.platform == 'linux':
        path = '/usr/local/cuda/version.txt'
        if os.path.isfile(path):
            with open(path, 'r') as f:
                data = f.read().replace('\n','')
            return data
        else:
            return "No CUDA in this machine"
    elif sys.platform == 'darwin':
        raise NotImplementedError("Find a Mac with GPU and implement this!")
    else:
        raise ValueError("Not in Windows, Linux or Mac")
        
    
def format_dictionary(dct, indent=4):
    """Formats a dictionary to be printed
    Parameters:
        dct (dict): Dictionary.
        indent (int): Indentation value.
    Returns:
        result (str): Formatted dictionary ready to be printed
    Examples:
        >>> dct = {'bkey':1, 'akey':2}
        >>> print(format_dictionary(dct))
        {
            "akey": 2,
            "bkey": 1
        }
    """
    return json.dumps(dct, indent=indent, sort_keys=True)



def get_filenames_in_folder(folderpath):
    """ Return the files names in a folder.
    Parameters:
        folderpath (str): folder path
    Returns:
        number (list): list of files
    Examples:
        >>> get_filenames_in_folder('C:/run3x/codebase/python/minsc')
        ['paths.py', 'system_info.py', '__init__.py']

    """
    names = [os.path.basename(x) for x in glob.glob(os.path.join(folderpath, '*'))]
    return sorted(names)


def get_files_in_folder_recursively(folderpath):
    """ Return the files inside a folder recursivaly.
    Parameters:
        folderpath (str): folder path
    Returns:
        filelist (list): list of files
    Examples:
        >>> get_files_in_folder_recursively(r'C:\\run3x\\codebase\\command_line')
        ['linux\\compress.txt', 'linux\\paths.txt', 'windows\\resources_management.txt']
    """
    if folderpath[-1] != os.path.sep: #Add final '/' if it doesn't exist
        folderpath += os.path.sep
    names = [x.replace(folderpath,'') for x in glob.iglob(folderpath+'/**', recursive=True) if os.path.isfile(x)]
    return sorted(names)



def _make_directory(directory):
    """Make a directory"""
    if not os.path.isdir(directory):
        os.makedirs(directory)
        

def decode_string(s):
    """Convert a string literal to a number or a bool.
    Parameters:
        s (str): String
    Returns:
        val (str,float,int or bool): Value decoded
    Examples:
        >>> decode_string('a')
        'a'
        >>> val = decode_string('1.0')
        >>> type(val)
        <class 'int'>
        >>> val
        1
        >>> val = decode_string('1')
        >>> type(val)
        <class 'int'>
        >>> val
        1
        >>> val = decode_string('1.5')
        >>> type(val)
        <class 'float'>
        >>> val
        1.5
        >>> val = decode_string('True')
        >>> type(val)
        <class 'bool'>
        >>> val
        True

    """
    if isinstance(s, str):
        # Does it represent a literal?
        try:
            val = literal_eval(s)
        except:
            # if it doesn't represent a literal, no conversion is done
            val = s
    else:
        # It's already something other than a string
        val = s
    # Is the float actually an int? (i.e. is the float 1.0 ?)
    if isinstance(val, float):
        if val.is_integer():
            return int(val)
        return val
    return val
