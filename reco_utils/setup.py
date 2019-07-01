# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from setuptools import setup, find_packages
from os import chdir, path


VERSION = __import__('__init__').VERSION

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

chdir(here)
setup(
    name='reco_utils',
    version=VERSION,
    description='Recommender System Utilities',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/microsoft/recommenders',
    author='RecoDev Team at Microsoft',
    author_email='RecoDevTeam@service.microsoft.com',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='recommendations recommenders recommender system engine machine learning python spark gpu',
    packages=find_packages(),
    python_requires='>=3.6, <4',
)
