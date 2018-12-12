import os
import random
import sys
from typing import List
from contextlib import ExitStack

def split_files(input_path: str, output_paths: List[str], ratios: List[float], max_rows=sys.maxsize):
    """Splits the input file into multiple output files according to the provided ratio.

    Splitting is performing in a streaming fashion to minimize memory consumption and thus be able 
    to operate on large files. Data is split stochastically, thus the files might not be exactly 
    split as requested by the ratios.
    
    Args:
        input_path (str): Path to input data file. Data is expected to be line separated.
        output_paths (List[str]): List of output filenames.
        ratios (List[float]): List of ratios used to split data.
        max_rows ([type], optional): Defaults to sys.maxsize. Maximum number of rows to be processed.
    
    Raises:
        ValueError: Number of output_paths must match ratios.
    """

    output_paths_len = len(output_paths)
    ratios_len = len(ratios)
    
    if output_paths_len == ratios_len + 1:
        ratios.append(1 - sum(ratios))
        ratios_len += 1
    elif output_paths_len != ratios_len:
        raise ValueError("Number of output_paths must match ratios")

    # get cumulative sum
    for i in range(1, ratios_len):
        ratios[i] = ratios[i] + ratios[i - 1]
    ratio_sum = ratios[-1]

    with ExitStack() as stack:
        output_files = [stack.enter_context(open(fname, 'w')) for fname in output_paths]

        for index, line in enumerate(open(input_path)):
            # limit number of rows
            if index >= max_rows:
                break

            p = random.random() * ratio_sum

            # safe guard against ratios not summing to 1
            idx = ratios_len - 1

            # find bucket
            for i, r in enumerate(ratios):
                if p < r:
                    idx = i
                    break

            output_files[idx].write(line)


def mkdir_safe(name: str):
    try:
        os.stat(name)
    except:
        os.mkdir(name)