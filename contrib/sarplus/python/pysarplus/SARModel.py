# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pysarplus_cpp
import os

from pathlib import Path


class SARModel:
    __path = None
    __model = None
    __extension = ".sar"

    def __init__(self, path):
        if SARModel.__model is not None and SARModel.__path == path:
            self.model = SARModel.__model
            return

        # find the .sar.related & .sar.offsets files
        sar_files = list(Path(path).glob("*" + SARModel.__extension))
        sar_files.sort(key=os.path.getmtime, reverse=True)
        if len(sar_files) < 1:
            raise ValueError(f"Directory '{path}' must contain at least 1 file ending in '{SARModel.__extension}'")

        # instantiate C++ backend
        SARModel.__model = self.model = pysarplus_cpp.SARModelCpp(str(sar_files[0]))
        SARModel.__path = path

    def predict(self, items, ratings, top_k, remove_seen):
        return self.model.predict(items, ratings, top_k, remove_seen)
