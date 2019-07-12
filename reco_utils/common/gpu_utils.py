# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import sys
import os
import glob
from numba import cuda
from numba.cuda.cudadrv.error import CudaSupportError


DEFAULT_CUDA_PATH_LINUX = "/usr/local/cuda/version.txt"


def get_number_gpus():
    """Get the number of GPUs in the system.
    
    Returns:
        int: Number of GPUs.
    """
    try:
        return len(cuda.gpus)
    except CudaSupportError:
        return 0


def get_gpu_info():
    """Get information of GPUs.

    Returns:
        list: List of gpu information dictionary  as `{device_name, total_memory (in Mb), free_memory (in Mb)}`.
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
        print("No CUDA available")


def get_cuda_version(unix_path=DEFAULT_CUDA_PATH_LINUX):
    """Get CUDA version.
    
    Args:
        unix_path (str): Path to CUDA version file in Linux/Mac.

    Returns:
        str: Version of the library.
    """
    if sys.platform == "win32":
        raise NotImplementedError("Implement this!")
    elif sys.platform in ["linux", "darwin"]:
        if os.path.isfile(unix_path):
            with open(unix_path, "r") as f:
                data = f.read().replace("\n", "")
            return data
        else:
            return "No CUDA in this machine"
    else:
        raise ValueError("Not in Windows, Linux or Mac")


def get_cudnn_version():
    """Get the CuDNN version.
    
    Returns:
        str: Version of the library.

    """

    def find_cudnn_in_headers(candidates):
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
            return "No CUDNN in this machine"

    if sys.platform == "win32":
        candidates = [
            "C:\\NVIDIA\\cuda\\include\\cudnn.h",
            "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\include\\cudnn.h",
        ]
    elif sys.platform == "linux":
        candidates = [
            "/usr/include/x86_64-linux-gnu/cudnn_v*.h",
            "/usr/local/cuda/include/cudnn.h",
            "/usr/include/cudnn.h",
        ]
    elif sys.platform == "darwin":
        candidates = ["/usr/local/cuda/include/cudnn.h", "/usr/include/cudnn.h"]
    else:
        raise ValueError("Not in Windows, Linux or Mac")
    return find_cudnn_in_headers(candidates)

