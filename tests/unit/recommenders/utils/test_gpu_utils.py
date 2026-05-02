# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import builtins
import sys
from types import SimpleNamespace
import pytest
import recommenders.utils.gpu_utils as gpu_utils

try:
    import tensorflow as tf
    import torch
    from recommenders.utils.gpu_utils import (
        get_cuda_version,
        get_cudnn_version,
        get_gpu_info,
        get_number_gpus,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment


def test_get_number_gpus_without_torch(monkeypatch):
    fake_cuda = SimpleNamespace(gpus=["gpu0", "gpu1"])
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "torch":
            raise ModuleNotFoundError("torch is unavailable in this test")
        if name == "numba":
            return SimpleNamespace()
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(gpu_utils, "cuda", fake_cuda)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert gpu_utils.get_number_gpus() == 2


@pytest.mark.gpu
def test_get_gpu_info():
    assert len(get_gpu_info()) >= 1


@pytest.mark.gpu
def test_get_number_gpus():
    assert get_number_gpus() >= 1


@pytest.mark.gpu
@pytest.mark.skip(reason="TODO: Implement this")
def test_clear_memory_all_gpus():
    pass


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform == "win32", reason="Not implemented on Windows")
def test_get_cuda_version():
    assert int(get_cuda_version().split(".")[0]) > 9


@pytest.mark.gpu
def test_get_cudnn_version():
    assert int(get_cudnn_version()[0]) > 7


@pytest.mark.gpu
def test_cudnn_enabled():
    assert torch.backends.cudnn.enabled is True


@pytest.mark.gpu
@pytest.mark.skip(reason="This function in TF is flaky")
def test_tensorflow_gpu():
    assert len(tf.config.list_physical_devices("GPU")) > 0


@pytest.mark.gpu
@pytest.mark.skip(reason="This function in PyTorch is flaky")
def test_pytorch_gpu():
    assert torch.cuda.is_available()
