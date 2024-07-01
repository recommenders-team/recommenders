# Copyright (c) Recommenders contributors.
# Licensed under the MIT license.

from enum import Enum


class LossFuncType(Enum):
    BCE = "bce"
    BPR = "bpr"
    SOFTMAX = "softmax"
    CCL = "ccl"
    FULLSOFTMAX = "fullsoftmax"


class DistanceType(Enum):
    DOT = "dot"
    COSINE = "cosine"
    MLP = "mlp"
