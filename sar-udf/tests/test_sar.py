import numpy as np
import unittest

import pysar


class TestSar(unittest.TestCase):
    def test_solve(self):
        x = pysar.SARModel("/Users/eisber/Documents/Recommenders/cpp-cache/output5")
        y = x.predict([0, 1], [10, 20], 10)

        for p in y:
            print("%d -> %f" % (p.id, p.score))
