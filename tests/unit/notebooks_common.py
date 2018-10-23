import os

# Unless manually modified, python3 should be the name of the current jupyter kernel
# that runs on the activated conda environment
KERNEL_NAME = "python3"
OUTPUT_NOTEBOOK = "output.ipynb"


def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, "notebooks"
        )
    )

