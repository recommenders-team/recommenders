# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import os
import glob
import logging
from numba import cuda
from numba.cuda.cudadrv.error import CudaSupportError


logger = logging.getLogger(__name__)


DEFAULT_CUDA_PATH_LINUX = "/usr/local/cuda/version.txt"


def get_number_gpus():
    """Get the number of GPUs in the system.
    
    Returns:
        int: Number of GPUs.
    """
    try:
        import torch
        return torch.cuda.device_count()
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import numba
        return len(numba.cuda.gpus)
    except Exception: # numba.cuda.cudadrv.error.CudaSupportError:
        return 0


def get_gpu_info():
    """Get information of GPUs.

    Returns:
        list: List of gpu information dictionary as with `device_name`, `total_memory` (in Mb) and `free_memory` (in Mb).
        Returns an empty list if there is no cuda device available.
    """
    gpus = []
    try:
        for gpu in cuda.gpus:
            with gpu:
                meminfo = cuda.current_context().get_memory_info()
                g = {
                    "device_name": gpu.name.decode("ASCII"),
                    "total_memory": meminfo[1] / 1048576,  # Mb
                    "free_memory": meminfo[0] / 1048576,  # Mb
                }
                gpus.append(g)
    except CudaSupportError:
        pass

    return gpus


def clear_memory_all_gpus():
    """Clear memory of all GPUs."""
    try:
        for gpu in cuda.gpus:
            with gpu:
                cuda.current_context().deallocations.clear()
    except CudaSupportError:
        logger.info("No CUDA available")


def get_cuda_version():
    """Get CUDA version
    
    Returns:
        str: Version of the library.
    """
    try:
        import torch
        return torch.version.cuda
    except (ImportError, ModuleNotFoundError):
        path = ""
        if sys.platform == "win32":
            candidate = (
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\version.txt"
            )
            path_list = glob.glob(candidate)
            if path_list:
                path = path_list[0]
        elif sys.platform == "linux" or sys.platform == "darwin":
            path = "/usr/local/cuda/version.txt"
        else:
            raise ValueError("Not in Windows, Linux or Mac")

        if os.path.isfile(path):
            with open(path, "r") as f:
                data = f.read().replace("\n", "")
            return data
        else:
            return "Cannot find CUDA in this machine"


def get_cudnn_version():
    """Get the CuDNN version
    
    Returns:
        str: Version of the library.
    """

    def find_cudnn_in_headers(candiates):
        for c in candidates:
            file = glob.glob(c)
            if file:
                break
        if file:
            with open(file[0], "r") as f:
                version = ""
                for line in f:
                    if "#define CUDNN_MAJOR" in line:
                        version = line.split()[-1]
                    if "#define CUDNN_MINOR" in line:
                        version += "." + line.split()[-1]
                    if "#define CUDNN_PATCHLEVEL" in line:
                        version += "." + line.split()[-1]
            if version:
                return version
            else:
                return "Cannot find CUDNN version"
        else:
            return "Cannot find CUDNN version"
            
    try:
        import torch
        return torch.backends.cudnn.version()
    except (ImportError, ModuleNotFoundError):
        if sys.platform == "win32":
            candidates = [r"C:\NVIDIA\cuda\include\cudnn.h"]
        elif sys.platform == "linux":
            candidates = [
                "/usr/include/cudnn_version.h",
                "/usr/include/x86_64-linux-gnu/cudnn_v[0-99].h",
                "/usr/local/cuda/include/cudnn.h",
                "/usr/include/cudnn.h",
            ]
        elif sys.platform == "darwin":
            candidates = ["/usr/local/cuda/include/cudnn.h", "/usr/include/cudnn.h"]
        else:
            raise ValueError("Not in Windows, Linux or Mac")
        return find_cudnn_in_headers(candidates)
