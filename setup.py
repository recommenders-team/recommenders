# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import site
import sys
import time
from os import environ
from pathlib import Path

from setuptools import find_packages, setup

<<<<<<< HEAD
=======
# Workaround for enabling editable user pip installs
>>>>>>> 22ac9e25 (Add python 3.11)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    # requires numpy and pandas>1.6 removes DataFrame.append() which is used in scrapbook.models
    "pandas>1.5.2,<1.6",
=======
    "pandas>1.5.3,<3.0.0",  # requires numpy
>>>>>>> a7f8346b (Update dep versions)
=======
    "pandas>2.0.0,<3.0.0",  # requires numpy
>>>>>>> 9f9c8153 (Fix pandas import)
    "scikit-learn>=1.1.3,<2",  # requires scipy
    "numba>=0.57.0,<1",
=======
    "category-encoders>=2.6.0,<3",
    "cornac>=1.15.2,<2",  # requires tqdm
    "hyperopt>=0.2.7,<1",
    "jinja2>=3.1.0,<3.2",
>>>>>>> 2fdf5901 (Set scipy <1.11.0 and sort dependencies alphabetically)
    "lightfm>=1.17,<2",
=======
    "category-encoders>=2.6.0,<3",  # requires packaging
    "cornac>=1.15.2,<2",  # requires packaging, tqdm
    "hyperopt>=0.2.7,<1",
    "lightfm>=1.17,<2",  # requires requests
>>>>>>> d249bfe6 (Remove duplicate dependencies jinja2 and packaging required other packages)
    "lightgbm>=4.0.0,<5",
    "locust>=2.12.2,<3",  # requires jinja2
    "memory-profiler>=0.61.0,<1",
    "nltk>=3.8.1,<4",  # requires tqdm
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    "seaborn>=0.12.0,<1",  # requires matplotlib
<<<<<<< HEAD
    "transformers>=4.26.0,<5",  # requires pyyaml, tqdm
    "bottleneck>=1.3.7,<2",
=======
=======
    "seaborn>=0.13.0,<1",  # requires matplotlib
>>>>>>> a7f8346b (Update dep versions)
    "transformers>=4.27.0,<5",  # requires pyyaml, tqdm
>>>>>>> 40361f4b (Fixed error: 'DataFrame' object has no attribute 'append')
    "category-encoders>=2.6.0,<3",
    "jinja2>=3.1.0,<3.2",
    "cornac>=1.15.2,<2",  # requires tqdm
<<<<<<< HEAD
    "retrying>=1.3.4",
    "pandera[strategies]>=0.15.0",  # For generating fake datasets
=======
    "retrying>=1.3.4,<2",
    "pandera[strategies]>=0.6.5,<0.18;python_version<='3.8'",  # For generating fake datasets
    "pandera[strategies]>=0.15.0;python_version>='3.9'",
>>>>>>> a7f8346b (Update dep versions)
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
=======
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
    "pandera[strategies]>=0.6.5,<0.18;python_version<='3.8'",  # For generating fake datasets
    "pandera[strategies]>=0.6.5;python_version>='3.9'",
    "scikit-surprise>=1.0.6",
    "hyperopt>=0.1.2,<1",
    "ipykernel>=4.6.1,<7",
    "jupyter>=1,<2",
    "locust>=1,<2",
>>>>>>> f5a15c83 (Fix pandera in Python 3.7)
=======
    "notebook>=7.0.0,<8",  # requires jupyter, ipykernel
=======
    "notebook>=7.0.0,<8",  # requires ipykernel, jinja2, jupyter, nbconvert, nbformat, packaging, requests
>>>>>>> d249bfe6 (Remove duplicate dependencies jinja2 and packaging required other packages)
    "numba>=0.57.0,<1",
    "pandas>2.0.0,<3.0.0",  # requires numpy
    "pandera[strategies]>=0.6.5,<0.18;python_version<='3.8'",  # For generating fake datasets
    "pandera[strategies]>=0.15.0;python_version>='3.9'",
    "retrying>=1.3.4,<2",
    "scikit-learn>=1.2.0,<2",  # requires scipy, and introduce breaking change affects feature_extraction.text.TfidfVectorizer.min_df
    "scikit-surprise>=1.1.3",
<<<<<<< HEAD
    "scipy>=1.7.2,<1.11.0",  # FIXME: We limit <1.11.0 until #1954 is fixed
    "seaborn>=0.13.0,<1",  # requires matplotlib
    "transformers>=4.27.0,<5",  # requires pyyaml, tqdm
>>>>>>> 2fdf5901 (Set scipy <1.11.0 and sort dependencies alphabetically)
=======
    "scipy>=1.10.1,<1.11.0",  # FIXME: We limit <1.11.0 until #1954 is fixed
    "seaborn>=0.13.0,<1",  # requires matplotlib, packaging
    "transformers>=4.27.0,<5",  # requires packaging, pyyaml, requests, tqdm
>>>>>>> d249bfe6 (Remove duplicate dependencies jinja2 and packaging required other packages)
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        "nvidia-ml-py>=11.510.69",
        # TensorFlow compiled with CUDA 11.8, cudnn 8.6.0.163
        "tensorflow~=2.12.0",
=======
        "nvidia-ml-py3>=7.352.0",
        "tensorflow>=2.8.4,!=2.9.0.*,!=2.9.1,!=2.9.2,!=2.10.0.*,<3",
>>>>>>> c736241b (Resolve issue #2018 (#2022))
        "tf-slim>=1.1.0",
        "torch>=2.0.1",
=======
=======
        "fastai>=2.7.11,<3",
>>>>>>> 2fdf5901 (Set scipy <1.11.0 and sort dependencies alphabetically)
        "nvidia-ml-py>=11.525.84",
<<<<<<< HEAD
        "tensorflow>=2.8.4,!=2.9.0.*,!=2.9.1,!=2.9.2,!=2.10.0.*,<2.16",
=======
        "fastai>=2.7.11,<3",
        "nvidia-ml-py>=11.525.84",
        "tensorflow>=2.8.4,!=2.9.0.*,!=2.9.1,!=2.9.2,!=2.10.0.*,<=2.15.0",
>>>>>>> 03554def (Set tensorflow <= 2.15.0)
=======
        "tensorflow>=2.8.4,!=2.9.0.*,!=2.9.1,!=2.9.2,!=2.10.0.*,<2.16",  # Fixed TF due to constant security problems and breaking changes #2073
>>>>>>> b255fae9 (:memo:)
        "tf-slim>=1.1.0",  # No python_requires in its setup.py
        "torch>=2.0.1,<3",
<<<<<<< HEAD
>>>>>>> a7f8346b (Update dep versions)
        "fastai>=2.7.11,<3",
=======
>>>>>>> 2fdf5901 (Set scipy <1.11.0 and sort dependencies alphabetically)
    ],
    "spark": [
        "databricks-cli>=0.17.7,<1",
        "pyarrow>=10.0.1",
        "pyspark>=3.3.0,<=4",
    ],
    "dev": [
<<<<<<< HEAD
<<<<<<< HEAD
        "black>=23.3.0,<24",
        "pytest>=7.2.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.10.0",  # for access to mock fixtures in pytest
        "pytest-rerunfailures>=11.1.2",  # to mark flaky tests
=======
        "black>=18.6b4,<21",
        "pytest>=3.6.4",
        "pytest-cov>=2.12.1",
        "pytest-mock>=3.6.1",  # for access to mock fixtures in pytest
        "packaging>=20.9",     # for version comparison in test_dependency_security.py
>>>>>>> c736241b (Resolve issue #2018 (#2022))
=======
        "black>=23.3.0",
        "pytest>=7.2.1",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.10.0",  # for access to mock fixtures in pytest
<<<<<<< HEAD
        "packaging>=22.0",     # for version comparison in test_dependency_security.py
>>>>>>> a7f8346b (Update dep versions)
=======
>>>>>>> 2fdf5901 (Set scipy <1.11.0 and sort dependencies alphabetically)
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
        "Documentation": "https://recommenders-team.github.io/recommenders/intro.html",
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
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        "Programming Language :: Python :: 3.11",
=======
>>>>>>> b71c4ed6 (Use docker images for ubuntu 22.04)
        "Operating System :: Microsoft :: Windows",
=======
        "Programming Language :: Python :: 3.11",
>>>>>>> 22ac9e25 (Add python 3.11)
=======
>>>>>>> c3a70302 (Remove python 3.11)
=======
        "Programming Language :: Python :: 3.11",
>>>>>>> 547ab666 (Try Python 3.11)
        "Operating System :: POSIX :: Linux",
    ],
    extras_require=extras_require,
    keywords="recommendations recommendation recommenders recommender system engine "
    "machine learning python spark gpu",
    install_requires=install_requires,
    package_dir={"recommenders": "recommenders"},
    python_requires=">=3.8, <=3.10",
    packages=find_packages(
        where=".",
        exclude=["contrib", "docs", "examples", "scenarios", "tests", "tools"],
    ),
    setup_requires=["numpy>=1.15"],
)
