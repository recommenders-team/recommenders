# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import sys
from types import SimpleNamespace
import pytest

try:
    import tensorflow as tf
    import torch
    import recommenders.utils.gpu_utils as gpu_utils
    from recommenders.utils.gpu_utils import (
        get_cuda_version,
        get_cudnn_version,
        get_gpu_info,
        get_number_gpus,
    )
except ImportError:
    pass  # skip this import if we are in cpu environment


def test_get_gpu_info_uses_torch_cuda(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            return 1

        @staticmethod
        def mem_get_info(device):
            assert device == 0
            return 256 * 1048576, 1024 * 1048576

        @staticmethod
        def get_device_name(device):
            assert device == 0
            return "Test GPU"

    monkeypatch.setattr(gpu_utils, "torch", SimpleNamespace(cuda=FakeCuda))

    assert gpu_utils.get_gpu_info() == [
        {
            "device_name": "Test GPU",
            "total_memory": 1024,
            "free_memory": 256,
        }
    ]


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
