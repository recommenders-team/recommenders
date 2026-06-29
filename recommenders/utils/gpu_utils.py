# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import logging

import torch

logger = logging.getLogger(__name__)


DEFAULT_CUDA_PATH_LINUX = "/usr/local/cuda/version.txt"


def get_number_gpus():
    """Get the number of GPUs in the system.
    Returns:
        int: Number of GPUs.
    """
    return torch.cuda.device_count()


def get_gpu_info():
    """Get information of GPUs.

    Returns:
        list: List of gpu information dictionary as with `device_name`, `total_memory` (in Mb) and `free_memory` (in Mb).
        Returns an empty list if there is no cuda device available.
    """
    gpus = []
    if torch.cuda.is_available():
        for device_id in range(torch.cuda.device_count()):
            free_memory, total_memory = torch.cuda.mem_get_info(device_id)
            gpus.append(
                {
                    "device_name": torch.cuda.get_device_name(device_id),
                    "total_memory": total_memory / 1048576,  # Mb
                    "free_memory": free_memory / 1048576,  # Mb
                }
            )
    return gpus


def clear_memory_all_gpus():
    """Clear memory of all GPUs."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        logger.info("No CUDA available")


def get_cuda_version():
    """Get CUDA version

    Returns:
        str: Version of the library.
    """
    return torch.version.cuda


def get_cudnn_version():
    """Get the CuDNN version

    Returns:
        str: Version of the library.
    """
    return str(torch.backends.cudnn.version())
