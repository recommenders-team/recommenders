# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

from pathlib import Path

from .SARModel import SARModel
from .SARPlus import SARPlus

__title__ = "pysarplus"
__version__ = (Path(__file__).resolve().parent / "VERSION").read_text().strip()
__author__ = "RecoDev Team at Microsoft"
__license__ = "MIT"
__copyright__ = "Copyright 2018-present Recommenders contributors."

# Synonyms
TITLE = __title__
VERSION = __version__
AUTHOR = __author__
LICENSE = __license__
COPYRIGHT = __copyright__
