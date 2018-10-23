import os

OUTPUT_NOTEBOOK = "output.ipynb"
KERNEL_NAME = "python3"


def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, "notebooks"
        )
    )

