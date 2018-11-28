# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.common.gpu_utils import get_number_gpus


@pytest.mark.notebooks
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1
