# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from setuptools import setup, find_packages
import time
from os import environ

# Version
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

name = environ.get("LIBRARY_NAME", "ms-recommenders")

install_requires = [
    "numpy>=1.14",
    "pandas>1.0.3,<2",
    "scipy>=1.0.0,<2",
    "tqdm>=4.31.1,<5",
    "matplotlib>=2.2.2,<4",
    "scikit-learn>=0.22.1,<1",
    "numba>=0.38.1,<1",
    "lightfm>=1.15,<2",
    "lightgbm>=2.2.1,<3",
    "memory_profiler>=0.54.0,<1",
    "nltk>=3.4,<4",
    "pydocumentdb>=2.3.3<3",  # TODO: replace with azure-cosmos
    "pymanopt>=0.2.5,<1",
    "seaborn>=0.8.1,<1",
    "transformers>=2.5.0,<5",
    "bottleneck>=1.2.1,<2",
    "category_encoders>=1.3.0,<2",
    "jinja2>=2,<3",
    "pyyaml>=5.4.1,<6",
    "requests>=2.0.0,<3",
    "cornac>=1.1.2,<2",
    "scikit-surprise>=0.19.1,<=1.1.1",
    "retrying>=1.3.3",
]

# shared dependencies
extras_require = {
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
        "tensorflow-gpu>=1.15.0,<2",  # compiled with CUDA 10.0
        "torch==1.2.0",  # last os-common version with CUDA 10.0 support
        "fastai>=1.0.46,<2",
    ],
    "spark": [
        "databricks_cli>=0.8.6,<1",
        "pyarrow>=0.8.0,<1.0.0",
        "pyspark>=2.4.5,<3.0.0",
    ],
    "xlearn": [
        "cmake>=3.18.4.post1",
        "xlearn==0.40a1",
    ],
    "dev": ["black>=18.6b4,<21", "pytest>=3.6.4"],
}
# for the brave of heart
extras_require["all"] = list(set(sum([*extras_require.values()], [])))

# the following dependencies need additional testing
extras_require["experimental"] = [
    "vowpalwabbit>=8.9.0,<9",
    "nni==1.5",
]


setup(
    name=name,
    version=version,
    description="Microsoft Recommenders - Python utilities for building recommender systems",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/microsoft/recommenders",
    project_urls={
        "Documentation": "https://microsoft-recommenders.readthedocs.io/en/stable/",
        "Wiki": "https://github.com/microsoft/recommenders/wiki",
    },
    author="RecoDev Team at Microsoft",
    author_email="RecoDevTeam@service.microsoft.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    extras_require=extras_require,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"reco_utils": "reco_utils"},
    packages=find_packages(where=".", exclude=["tests", "tools", "examples"]),
    python_requires=">=3.6, <=3.7.10",
)
