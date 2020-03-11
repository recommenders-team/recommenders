# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import time
from setuptools import setup, find_packages
from os import chdir, path, environ

chdir(path.abspath(path.dirname(__file__)))
version = __import__("reco_utils.__init__").VERSION

# Get the long description from the README file
with open(path.join("reco_utils", "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

HASH = environ.get("HASH", None)
if HASH is not None:
    version += ".dev" + str(int(time.time()))

setup(
    name="reco_utils",
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
    keywords="recommendations recommenders recommender system engine machine learning python spark gpu",
    package_dir={"": "reco_utils"},
    packages=find_packages(where="reco_utils", exclude=["azureml_designer_modules"]),
    python_requires=">=3.6, <4",
)
