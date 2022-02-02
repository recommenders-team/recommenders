# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from os import environ
from pathlib import Path
from setuptools import setup, find_packages
import site
import sys
import time

# workround for enabling editable user pip installs
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# version
here = Path(__file__).absolute().parent
version_data = {}
with open(here.joinpath("recommenders", "__init__.py"), "r") as f:
    exec(f.read(), version_data)
version = version_data.get("__version__", "0.0")

# Get the long description from the README file
with open(here.joinpath("recommenders", "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

HASH = environ.get("HASH", None)
if HASH is not None:
    version += ".post" + str(int(time.time()))

install_requires = [
    "numpy>=1.19",  # 1.19 required by tensorflow 2.6
    "pandas>1.0.3,<2",
    "scipy>=1.0.0,<2",
    "tqdm>=4.31.1,<5",
    "matplotlib>=2.2.2,<4",
    "scikit-learn>=0.22.1,<1",
    "numba>=0.38.1,<1",
    "lightfm>=1.15,<2",
    "lightgbm>=2.2.1",
    "memory_profiler>=0.54.0,<1",
    "nltk>=3.4,<4",
    "pydocumentdb>=2.3.3<3",  # TODO: replace with azure-cosmos
    "seaborn>=0.8.1,<1",
    "transformers>=2.5.0,<5",
    "bottleneck>=1.2.1,<2",
    "category_encoders>=1.3.0,<2",
    "jinja2>=2,<4",
    "pyyaml>=5.4.1,<6",
    "requests>=2.0.0,<3",
    "cornac>=1.1.2,<2",
    "retrying>=1.3.3",
    "pandera[strategies]>=0.6.5",  # For generating fake datasets
]

# shared dependencies
extras_require = {
    "examples": [
        "azure.mgmt.cosmosdb>=0.8.0,<1",
        "hyperopt>=0.1.2,<1",
        "ipykernel>=4.6.1,<7",
        "jupyter>=1,<2",
        "locust>=1,<2",
        "papermill>=2.1.2,<3",
        "scrapbook>=0.5.0,<1.0.0",
    ],
    "gpu": [
        "nvidia-ml-py3>=7.352.0",
        # TensorFlow compiled with CUDA 11.2, cudnn 8.1
        "tensorflow~=2.6.1;python_version=='3.6'",
        "tensorflow~=2.7.0;python_version>='3.7'",
        "tf-slim>=1.1.0",
        "torch>=1.8",  # for CUDA 11 support
        "fastai>=1.0.46,<2",
    ],
    "spark": [
        "databricks_cli>=0.8.6,<1",
        "pyarrow>=0.12.1,<7.0.0",
        "pyspark>=2.4.5,<4.0.0",
    ],
    "dev": [
        "black>=18.6b4,<21",
        "pytest>=3.6.4",
        "pytest-cov>=2.12.1",
        "pytest-mock>=3.6.1",  # for access to mock fixtures in pytest
        "pytest-rerunfailures>=10.2",  # to mark flaky tests
    ],
}
# for the brave of heart
extras_require["all"] = list(set(sum([*extras_require.values()], [])))

# the following dependencies need additional testing
extras_require["experimental"] = [
    # xlearn requires cmake to be pre-installed
    "xlearn==0.40a1",
    # VW C++ binary needs to be installed manually for some code to work
    "vowpalwabbit>=8.9.0,<9",
]
extras_require["nni"] = [
    # nni needs to be upgraded
    "nni==1.5",
]

# The following dependencies can be installed as below, however PyPI does not allow direct URLs.
# Surprise needs to be built from source because of the numpy <= 1.19 incompatibility
# Requires pip to be run with the --no-binary option
# "scikit-surprise@https://github.com/NicolasHug/Surprise/archive/refs/tags/v1.1.1.tar.gz",
# Temporary fix for pymanopt, only this commit works with TF2
# "pymanopt@https://github.com/pymanopt/pymanopt/archive/fb36a272cdeecb21992cfd9271eb82baafeb316d.zip",

setup(
    name="recommenders",
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
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    extras_require=extras_require,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"recommenders": "recommenders"},
    python_requires=">=3.6, <3.10",
    packages=find_packages(where=".", exclude=["contrib", "docs", "examples", "scenarios", "tests", "tools"]),
    setup_requires=["numpy>=1.15"]
)
