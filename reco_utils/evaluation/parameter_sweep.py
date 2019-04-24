# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#
# Utility functions for parameter sweep.

from itertools import product


def generate_param_grid(params):
    """Generator of parameter grids
    Generate parameter lists from a parameter dictionary in the form of
    {
        "param1": [value1, value2],
        "param2": [value1, value2]
    }

    to

    [
        {"param1": value1, "param2": value1},
        {"param1": value2, "param2": value1},
        {"param1": value1, "param2": value2},
        {"param1": value2, "param2": value2}
    ]

    Args:
        param_dict (dict): dictionary of parameters and values (in a list).

    Return:
        list: A list of parameter dictionary string that can be fed directly into
        model builder as keyword arguments.
    """
    param_new = {}
    param_fixed = {}

    for key, value in params.items():
        if isinstance(value, list):
            param_new[key] = value
        else:
            param_fixed[key] = value

    items = sorted(param_new.items())
    keys, values = zip(*items)

    params_exp = []
    for v in product(*values):
        param_exp = dict(zip(keys, v))
        param_exp.update(param_fixed)
        params_exp.append(param_exp)

    return params_exp

