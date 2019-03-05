# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


def invert_dictionary(dictionary):
    """Invert a dictionary
    NOTE: If the dictionary has unique keys and unique values, the invertion would be perfect. However, if there are
    repeated values, the invertion can take different keys
    
    Args: 
        dictionary (dict): A dictionary
    
    Returns:
        dict: inverted dictionary
    """
    return {v: k for k, v in dictionary.items()}


def get_number_processors():
    """Get the number of processors in a CPU.
    
    Returns:
        int: Number of processors.
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing  # force exception in case mutiprocessing is not installed

        num = multiprocessing.cpu_count()
    return num
