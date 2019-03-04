# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


def invert_dictionary(dictionary):
    """Invert a dictionary
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
