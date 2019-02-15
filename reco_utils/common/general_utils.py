# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


def invert_dictionary(dictionary):
    """Invert a dictionary
    Args: 
        dictionary (dict): A dictionary
    Returns:
        dict: inverted dictionary
    """
    return {v: k for k, v in dictionary.items()}
