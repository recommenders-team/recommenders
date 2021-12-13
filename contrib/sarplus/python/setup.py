# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sysconfig

from pathlib import Path
from setuptools import setup
from setuptools.extension import Extension


class get_pybind_include(object):
    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11

        return pybind11.get_include(self.user)


DEPENDENCIES = [
    "numpy",
    "pandas",
    "pyarrow>=1.0.0",
    "pybind11>=2.2",
    "pyspark>=3.0.0"
]

setup(
    name="pysarplus",
    version=(Path(__file__).resolve().parent / "VERSION").read_text().strip(),
    description="SAR prediction for use with PySpark",
    long_description=(Path(__file__).resolve().parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    url="https://github.com/microsoft/recommenders/tree/main/contrib/sarplus",
    author="RecoDev Team at Microsoft",
    author_email="recodevteam@service.microsoft.com",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    setup_requires=["pytest-runner"],
    install_requires=DEPENDENCIES,
    tests_require=["pytest"],
    packages=["pysarplus"],
    ext_modules=[
        Extension(
            "pysarplus_cpp",
            ["src/pysarplus.cpp"],
            include_dirs=[get_pybind_include(), get_pybind_include(user=True)],
            extra_compile_args=sysconfig.get_config_var("CFLAGS").split() + ["-std=c++11", "-Wall", "-Wextra"],
            libraries=["stdc++"],
            language="c++11",
        )
    ],
    zip_safe=False,
)
