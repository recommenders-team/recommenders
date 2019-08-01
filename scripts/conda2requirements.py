# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# This file parses a conda file and outputs a requirements.txt file with the same libraries

import argparse
import yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converter from conda.yml to requirements.txt",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", help="Input filename", default="conda.yaml")
    parser.add_argument(
        "-o", "--output", help="Output filename", default="requirements.txt"
    )
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = yaml.load(f, yaml.SafeLoader)

    conda_deps = [x for x in data["dependencies"] if isinstance(x, str)]

    pip_deps = [x for x in data["dependencies"] if isinstance(x, dict)]
    pip_deps = pip_deps[0]["pip"]

    deps = conda_deps + pip_deps

    with open(args.output, "w") as f:
        f.write("\n".join(deps))
