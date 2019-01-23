# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from reco_utils.common.gpu_utils import get_number_gpus
from tests.notebooks_common import OUTPUT_NOTEBOOK, KERNEL_NAME
import papermill as pm

@pytest.mark.notebooks
@pytest.mark.gpu
def test_gpu_vm():
    assert get_number_gpus() >= 1


@pytest.mark.notebooks
@pytest.mark.gpu
def test_fastai(notebooks):
    assert get_number_gpus() >= 1

    notebook_path = notebooks["fastai"]
    print('running fastai unit test')
    pm.execute_notebook(notebook_path, OUTPUT_NOTEBOOK, kernel_name=KERNEL_NAME)