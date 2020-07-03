# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file outputs a requirements.txt based on the libraries defined in generate_conda_file.py
from generate_conda_file import (
    CONDA_BASE,
    CONDA_PYSPARK,
    CONDA_GPU,
    PIP_BASE,
    PIP_GPU,
    PIP_PYSPARK,
    PIP_DARWIN,
    PIP_LINUX,
    PIP_WIN32,
)


if __name__ == "__main__":
    deps = list(CONDA_BASE.values())
    deps += list(CONDA_PYSPARK.values())
    deps += list(CONDA_GPU.values())
    deps += list(PIP_BASE.values())
    deps += list(PIP_PYSPARK.values())
    deps += list(PIP_GPU.values())
    deps += list(PIP_DARWIN.values())
    deps += list(PIP_LINUX.values())
    deps += list(PIP_WIN32.values())
    with open("requirements.txt", "w") as f:
        f.write("\n".join(set(deps)))
