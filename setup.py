# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from setuptools import setup, find_packages
import time
from os import environ

here = Path(__file__).absolute().parent
version_data = {}
with open(here.joinpath("reco_utils", "__init__.py"), "r") as f:
    exec(f.read(), version_data)
version = version_data.get("__version__", "0.0")

# Get the long description from the README file
with open(here.joinpath("reco_utils", "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

HASH = environ.get("HASH", None)
if HASH is not None:
    version += ".post" + str(int(time.time()))

name = environ.get("LIBRARY_NAME", "reco_utils")

install_requires = [
    "matplotlib>=2.2.2,<3",
    "memory_profiler>=0.54.0,<1",
    "nni==1.5",
    "numba>=0.38.1,<1",
    "numpy>=1.13.3,<2",
    "pandas>1.0.3,<2",
    "pydocumentdb>=2.3.3<3",  # todo: replace with azure-cosmos
    "pyyaml>=5.4.1,<6",
    "requests>=2.0.0,<3",
    "seaborn>=0.8.1,<1",
    "scipy>=1.0.0,<2",
    "scikit-learn>=0.19.1,<1",
    "tqdm>=4.31.1,<5",
]

extras_require = {
    "recommenders": [
        "bottleneck>=1.2.1,<2",
        "category_encoders>=1.3.0,<2",
        "cornac>=1.1.2,<2",
        "jinja2>=2,<3",
        "lightfm>=1.15,<2",
        "lightgbm>=2.2.1,<3",
        "nltk>=3.4,<4",
        "pymanopt>=0.2.5,<1",
        "scikit-surprise>=0.19.1,<2",
        "transformers>=2.5.0,<5",
    ],
    "examples": [
        "azure.mgmt.cosmosdb>=0.8.0,<1",
        "hyperopt>=0.1.2,<1",
        "ipykernel>=4.6.1,<5",
        "jupyter>=1,<2",
        "locust>=1,<2",
        "papermill>=2.1.2,<3",
        "scrapbook>=0.5.0,<1.0.0",        
    ],
    "gpu": [
        "nvidia-ml-py3>=7.352.0",
        "tensorflow-gpu==1.15.4",
        "fastai>=1.0.46,<2",
        "torch>=1.0.0,<2",
    ],
    "spark": [
        "databricks_cli>=0.8.6,<1",
        "pyarrow>=0.8.0,<1.0.0",
        "pyspark>=2.4.5,<3.0.0",
    ],
    "test": [
        "black>=18.6b4,<21",
        "papermill>=2.1.2,<3",
        "pytest>=3.6.4",
        "scrapbook>=0.5.0,<1.0.0",
    ],
}
# for the brave of heart
extras_require["all"] = list(set(sum([*extras_require.values()], [])))

# the following dependencies need additional testing
extras_require["beta"] = [
    "azureml-sdk[notebooks,tensorboard]>=1.0.69,<2",
    "xlearn==0.40a1",
    "vowpal_wabbit>=8.9.0,<9",
]


setup(
    name=name,
    version=version,
    description="Recommender System Utilities",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/recommenders",
    author="RecoDev Team at Microsoft",
    author_email="RecoDevTeam@service.microsoft.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
    extras_require=extras_require,
    keywords="recommendations recommenders recommender system engine machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"reco_utils": "reco_utils"},
    packages=find_packages(where=".", exclude=["tests", "tools", "examples"]),
    python_requires=">=3.6, <4",
)
