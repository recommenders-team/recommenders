# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import os

import numpy as np
import pytest

FFM_FIELD_COUNT = 6
FFM_SPLITS = [("train", 40), ("valid", 20), ("test", 20)]


@pytest.fixture(scope="module")
def synthetic_ffm(tmp_path_factory):
    """Tiny synthetic FFM files, so the tests need no download.

    Every line carries exactly one feature per field, with the 1-based field and
    feature indices the format uses, and alternating 0/1 labels.
    """
    d = tmp_path_factory.mktemp("ffm")
    rng = np.random.RandomState(0)
    paths = {}
    for name, n_lines in FFM_SPLITS:
        path = os.path.join(d, name)
        with open(path, "w") as f:
            for i in range(n_lines):
                parts = [str(i % 2)]
                for field in range(1, FFM_FIELD_COUNT + 1):
                    feature = field * 20 + rng.randint(1, 11)
                    parts.append("{0}:{1}:1".format(field, feature))
                f.write(" ".join(parts) + "\n")
        paths[name] = path
    return paths
