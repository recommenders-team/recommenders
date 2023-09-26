# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.


import pytest


@pytest.mark.gpu
def test_compatibility_tf():
    """Some of our code uses TF1 and some TF2. Here we just check that we
    can import both versions.
    """
    import tensorflow as tf
    from tensorflow.compat.v1 import placeholder
