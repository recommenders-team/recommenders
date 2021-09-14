# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os


def is_jupyter():
    """Check if the module is running on Jupyter notebook/console.

    Returns:
        bool: True if the module is running on Jupyter notebook or Jupyter console,
        False otherwise.
    """
    try:
        shell_name = get_ipython().__class__.__name__
        if shell_name == "ZMQInteractiveShell":
            return True
        else:
            return False
    except NameError:
        return False


def is_databricks():
    """Check if the module is running on Databricks.

    Returns:
        bool: True if the module is running on Databricks notebook,
        False otherwise.
    """
    try:
        if os.path.realpath(".") == "/databricks/driver":
            return True
        else:
            return False
    except NameError:
        return False
