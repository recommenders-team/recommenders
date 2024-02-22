# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import re
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from IPython.display import display


NOTEBOOK_OUTPUT_CONTENT_TYPE = "application/notebook_utils.json+json"


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


def _update_parameters(parameter_cell_source, new_parameters):
    """Replace parameter values in the cell source code."""
    modified_cell_source = parameter_cell_source
    for param, new_value in new_parameters.items():
        if (
            isinstance(new_value, str)
            and not (new_value.startswith('"') and new_value.endswith('"'))
            and not (new_value.startswith("'") and new_value.endswith("'"))
        ):
            # Check if the new value is a string and surround it with quotes if necessary
            new_value = f'"{new_value}"'

        # Define a regular expression pattern to match parameter assignments and ignore comments
        pattern = re.compile(rf"(\b{param})\s*=\s*([^#\n]+)(?:#.*$)?", re.MULTILINE)
        modified_cell_source = pattern.sub(rf"\1 = {new_value}", modified_cell_source)

    return modified_cell_source


def execute_notebook(
    input_notebook, output_notebook, parameters={}, kernel_name="python3", timeout=2200
):
    """Execute a notebook while passing parameters to it.

    Note:
        Ensure your Jupyter Notebook is set up with parameters that can be
        modified and read. Use Markdown cells to specify parameters that need
        modification and code cells to set parameters that need to be read.

    Args:
        input_notebook (str): Path to the input notebook.
        output_notebook (str): Path to the output notebook
        parameters (dict): Dictionary of parameters to pass to the notebook.
        kernel_name (str): Kernel name.
        timeout (int): Timeout (in seconds) for each cell to execute.
    """

    # Load the Jupyter Notebook
    with open(input_notebook, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Search for and replace parameter values in code cells
    for cell in notebook_content.cells:
        if (
            "tags" in cell.metadata
            and "parameters" in cell.metadata["tags"]
            and cell.cell_type == "code"
        ):
            # Update the cell's source within notebook_content
            cell.source = _update_parameters(cell.source, parameters)

    # Create an execution preprocessor
    execute_preprocessor = ExecutePreprocessor(timeout=timeout, kernel_name=kernel_name)

    # Execute the notebook
    executed_notebook, _ = execute_preprocessor.preprocess(
        notebook_content, {"metadata": {"path": "./"}}
    )

    # Save the executed notebook
    with open(output_notebook, "w", encoding="utf-8") as executed_notebook_file:
        nbformat.write(executed_notebook, executed_notebook_file)


def store_metadata(name, value):
    """Store data in the notebook's output source code.

    Args:
        name (str): Name of the data.
        value (int,float,str): Value of the data.
    """

    metadata = {"notebook_utils": {"name": name, "data": True, "display": False}}
    data_json = {
        "application/notebook_utils.json+json": {
            "name": name,
            "data": value,
            "encoder": "json",
        }
    }
    display(data_json, metadata=metadata, raw=True)


def read_notebook(path):
    """Read the metadata stored in the notebook's output source code.

    Args:
        path (str): Path to the notebook.

    Returns:
        dict: Dictionary of data stored in the notebook.
    """
    # Load the Jupyter Notebook
    with open(path, "r") as notebook_file:
        notebook_content = nbformat.read(notebook_file, as_version=4)

    # Search for parameters and store them in a dictionary
    results = {}
    for cell in notebook_content.cells:
        if cell.cell_type == "code" and "outputs" in cell:
            for outputs in cell.outputs:
                if "metadata" in outputs and "notebook_utils" in outputs.metadata:
                    name = outputs.data[NOTEBOOK_OUTPUT_CONTENT_TYPE]["name"]
                    data = outputs.data[NOTEBOOK_OUTPUT_CONTENT_TYPE]["data"]
                    results[name] = data
    return results
