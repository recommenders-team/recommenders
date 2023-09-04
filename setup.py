# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import site
import sys
import time
from os import environ
from pathlib import Path

from setuptools import find_packages, setup

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
    # requires numpy and pandas>1.6 removes DataFrame.append() which is used in scrapbook.models
    "pandas>1.5.2,<1.6",
    "scikit-learn>=1.1.3,<2",  # requires scipy
    "numba>=0.57.0,<1",
    "lightfm>=1.17,<2",
    "lightgbm>=3.3.2,<5",
    "memory-profiler>=0.61.0,<1",
    "nltk>=3.8.1,<4",  # requires tqdm
    "seaborn>=0.12.0,<1",  # requires matplotlib
<<<<<<< HEAD
    "transformers>=4.26.0,<5",  # requires pyyaml, tqdm
    "bottleneck>=1.3.7,<2",
=======
    "transformers>=4.27.0,<5",  # requires pyyaml, tqdm
>>>>>>> 40361f4b (Fixed error: 'DataFrame' object has no attribute 'append')
    "category-encoders>=2.6.0,<3",
    "jinja2>=3.1.0,<3.2",
    "cornac>=1.15.2,<2",  # requires tqdm
    "retrying>=1.3.4",
    "pandera[strategies]>=0.15.0",  # For generating fake datasets
    "scikit-surprise>=1.1.3",
    "scrapbook>=0.5.0,<1.0.0",  # requires tqdm, papermill
<<<<<<< HEAD
=======
    "hyperopt>=0.2.7,<1",
    "notebook>=7.0.0,<8",  # requires jupyter, ipykernel
    "locust>=2.12.2,<3",
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> 40361f4b (Fixed error: 'DataFrame' object has no attribute 'append')
=======
    # 6.83.1 introduced a non-existent attribute '_deferred_pprinters' of IPython.lib.pretty in
=======
    # hypothesis 6.83.1 introduced a non-existent attribute '_deferred_pprinters' of IPython.lib.pretty in
>>>>>>> 0641d953 (Update comments)
    # https://github.com/HypothesisWorks/hypothesis/commit/5ea8e0c3e6da1cd9fb3f302124dc74791c14db11
    "hypothesis<6.83.1",
>>>>>>> 9364c9b7 (Add hypothesis<6.83.1)
]

# shared dependencies
extras_require = {
    "examples": [
        "azure-mgmt-cosmosdb>=9.0.0,<10",
        "hyperopt>=0.2.7,<1",
        "notebook>=6.5.4,<8",
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
        "databricks-cli>=0.17.7,<1",
        "pyarrow>=10.0.1",
        "pyspark>=3.0.1,<=3.4.0",
    ],
    "dev": [
        "black>=23.3.0,<24",
        "pytest>=7.2.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.10.0",  # for access to mock fixtures in pytest
        "pytest-rerunfailures>=11.1.2",  # to mark flaky tests
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
    description="Recommenders - Python utilities for building recommendation systems",
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
        "Programming Language :: Python :: 3.10",
<<<<<<< HEAD
        "Programming Language :: Python :: 3.11",
=======
>>>>>>> b71c4ed6 (Use docker images for ubuntu 22.04)
        "Operating System :: Microsoft :: Windows",
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
    setup_requires=["numpy>=1.15"],
)
