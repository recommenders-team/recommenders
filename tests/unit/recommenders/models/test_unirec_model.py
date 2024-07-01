# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os
import pytest

try:
    from recommenders.models.unirec.model.sequential.sasrec import SASRec
except ImportError:
    pass  # skip this import if we are in cpu environment


@pytest.mark.gpu
def test_sasrec_component_definition():
    model = SASRec()
