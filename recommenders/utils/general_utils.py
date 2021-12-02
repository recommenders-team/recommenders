# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import psutil


def invert_dictionary(dictionary):
    """Invert a dictionary.

    .. note::

        If the dictionary has unique keys and unique values, the inversion would be perfect. However, if there are
        repeated values, the inversion can take different keys

    Args:
        dictionary (dict): A dictionary

    Returns:
        dict: inverted dictionary
    """
    return {v: k for k, v in dictionary.items()}


def get_physical_memory():
    """Get the physical memory in GBs.

    Returns:
        float: Physical memory in GBs.
    """
    return psutil.virtual_memory()[0] / 1073741824


def get_number_processors():
    """Get the number of processors in a CPU.

    Returns:
        int: Number of processors.
    """
    try:
        num = os.cpu_count()
    except Exception:
        import multiprocessing  # force exception in case multiprocessing is not installed

        num = multiprocessing.cpu_count()
    return num
