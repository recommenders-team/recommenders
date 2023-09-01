# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from os import environ
from pathlib import Path
from setuptools import setup, find_packages
import site
import sys
import time

# workaround for enabling editable user pip installs
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
    "pandas>1.5.2,<2.1",  # requires numpy
    "scikit-learn>=1.1.3,<2",  # requires scipy
    "numba>=0.57.0,<1",
    "lightfm>=1.17,<2",
    "lightgbm>=3.3.2,<4",
    "memory-profiler>=0.61.0,<1",
    "nltk>=3.8.1,<4",  # requires tqdm
    "seaborn>=0.12.0,<1",  # requires matplotlib
    "transformers>=4.26.0,<5",  # requires pyyaml, tqdm
    "category-encoders>=2.6.0,<3",
    "jinja2>=3.1.0,<3.2",
    "cornac>=1.15.2,<2",  # requires tqdm
    "retrying>=1.3.4",
    "pandera[strategies]>=0.15.0",  # For generating fake datasets
    "scikit-surprise>=1.1.3",
    "scrapbook>=0.5.0,<1.0.0",  # requires tqdm, papermill
]

# shared dependencies
extras_require = {
    "examples": [
        "hyperopt>=0.2.7,<1",
        "notebook>=6.5.4,<8",  # requires jupyter, ipykernel
        "locust>=2.15.1,<3",
    ],
    "gpu": [
        "nvidia-ml-py>=11.510.69",
        # TensorFlow compiled with CUDA 11.8, cudnn 8.6.0.163
        "tensorflow~=2.12.0",
        "tf-slim>=1.1.0",
        "torch>=2.0.1",
        "fastai>=2.7.11,<3",
    ],
    "spark": [
        "pyarrow>=10.0.1",
        "pyspark>=3.0.1,<=3.4.0",
    ],
    "dev": [
        "black>=23.3.0,<24",
        "pytest>=7.2.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.10.0",  # for access to mock fixtures in pytest
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

# The following dependency can be installed as below, however PyPI does not allow direct URLs.
# Temporary fix for pymanopt, only this commit works with TF2
# "pymanopt@https://github.com/pymanopt/pymanopt/archive/fb36a272cdeecb21992cfd9271eb82baafeb316d.zip",

setup(
    name="recommenders",
    version=version,
    description="Recommenders - Python utilities for building recommender systems",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/recommenders-team/recommenders",
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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: POSIX :: Linux",
    ],
    extras_require=extras_require,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"recommenders": "recommenders"},
    python_requires=">=3.8, <3.11",
    packages=find_packages(
        where=".",
        exclude=["contrib", "docs", "examples", "scenarios", "tests", "tools"],
    ),
    setup_requires=["numpy>=1.19"],
)
