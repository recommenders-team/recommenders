# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from os import environ
from pathlib import Path
from setuptools import setup, find_packages
import site
import sys
import time

# Workaround for enabling editable user pip installs
site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

# Version
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
    "scipy>=1.0.0,<1.11.0",  # FIXME: We limit <1.11.0 until #1954 is fixed
    "tqdm>=4.31.1,<5",
    "matplotlib>=2.2.2,<4",
    "scikit-learn>=0.22.1,<1.0.3",
    "numba>=0.38.1,<1",
    "lightfm>=1.15,<2",
    "lightgbm>=2.2.1",
    "memory_profiler>=0.54.0,<1",
    "nltk>=3.4,<4",
    "seaborn>=0.8.1,<1",
    "transformers>=2.5.0,<5",
    "category_encoders>=1.3.0,<2",
    "jinja2>=2,<3.1",
    "requests>=2.31.0,<3",
    "cornac>=1.1.2,<1.15.2;python_version<='3.7'",
    "cornac>=1.15.2,<2;python_version>='3.8'",  # After 1.15.2, Cornac requires python 3.8
    "retrying>=1.3.3",
    "pandera[strategies]>=0.6.5",  # For generating fake datasets
    "scikit-surprise>=1.0.6",
    "scrapbook>=0.5.0,<1.0.0",
    "hyperopt>=0.1.2,<1",
    "ipykernel>=4.6.1,<7",
    "jupyter>=1,<2",
    "locust>=1,<2",
    "papermill>=2.1.2,<3",
]

# shared dependencies
extras_require = {
    "gpu": [
        "nvidia-ml-py3>=7.352.0",
        "tensorflow==2.8.4",  # FIXME: Temporarily pinned due to issue with TF version > 2.10.1 See #2018
        "tf-slim>=1.1.0",
        "torch>=1.13.1",  # for CUDA 11 support
        "fastai>=1.0.46,<2",
    ],
    "spark": [
        "pyarrow>=0.12.1,<7.0.0",
        "pyspark>=2.4.5,<3.3.0",
    ],
    "dev": [
        "black>=18.6b4,<21",
        "pytest>=3.6.4",
        "pytest-cov>=2.12.1",
        "pytest-mock>=3.6.1",  # for access to mock fixtures in pytest
    ],
}
# For the brave of heart
extras_require["all"] = list(set(sum([*extras_require.values()], [])))

# The following dependencies need additional testing
extras_require["experimental"] = [
    # xlearn requires cmake to be pre-installed
    "xlearn==0.40a1",
    # VW C++ binary needs to be installed manually for some code to work
    "vowpalwabbit>=8.9.0,<9",
    # nni needs to be upgraded
    "nni==1.5",
    "pymanopt>=0.2.5",
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
        "Wiki": "https://github.com/recommenders-team/recommenders/wiki",
    },
    author="Recommenders contributors",
    author_email="recommenders-technical-discuss@lists.lfaidata.foundation",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: POSIX :: Linux",
    ],
    extras_require=extras_require,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"recommenders": "recommenders"},
    python_requires=">=3.6",
    packages=find_packages(
        where=".",
        exclude=["contrib", "docs", "examples", "scenarios", "tests", "tools"],
    ),
    setup_requires=["numpy>=1.19"],
)
