# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from numba import cuda
from numba.cuda.cudadrv.error import CudaSupportError


def get_number_gpus():
    """Get the number of GPUs in the system.
    Returns:
        int: Number of GPUs.
    """
    try:
        return len(cuda.gpus)
    except CudaSupportError:
        return 0

    
def clear_memory_all_gpus():
    """Clear memory of all GPUs.
    """
    try:
        for gpu in cuda.gpus:
            with gpu:
                cuda.current_context().deallocations.clear()
    except CudaSupportError:
        print("No CUDA available")
        
