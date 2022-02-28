# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

from .SARModel import SARModel
from .SARPlus import SARPlus

__title__ = "pysarplus"
__version__ = (Path(__file__).resolve().parent / "VERSION").read_text().strip()
__author__ = "RecoDev Team at Microsoft"
__license__ = "MIT"
__copyright__ = "Copyright 2018-present Microsoft Corporation"

# Synonyms
TITLE = __title__
VERSION = __version__
AUTHOR = __author__
LICENSE = __license__
COPYRIGHT = __copyright__
