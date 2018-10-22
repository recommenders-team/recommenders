import os

OUTPUT_NOTEBOOK = "output.ipynb"


def path_notebooks():
    """Returns the path of the notebooks folder"""
    return os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), os.path.pardir, os.path.pardir, "notebooks"
        )
    )


def conda_environment_name():
    """Returns the current conda environment name"""
    return os.environ["CONDA_DEFAULT_ENV"]
