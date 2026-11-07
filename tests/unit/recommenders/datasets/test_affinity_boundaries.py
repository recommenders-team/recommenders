# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recommenders.datasets.sparse import AffinityMatrix


class TestAffinityBoundaries(unittest.TestCase):
    def make_matrix(self, **options):
        frame = pd.DataFrame({"userID": [7], "itemID": [19], "rating": [4]})
        return AffinityMatrix(frame, **options)

    def test_no_ratings(self):
        matrix = self.make_matrix(items_list=[19, 22])
        matrix.df = matrix.df.iloc[:0]
        values, users, items = matrix.gen_affinity_matrix()
        self.assertEqual(values.shape, (0, 2))
        self.assertEqual(users, {})
        self.assertEqual(items, {19: 0, 22: 1})

    def test_duplicate_catalog(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self.make_matrix(items_list=[19, 19]).gen_affinity_matrix()

    def test_unknown_item(self):
        with self.assertRaisesRegex(ValueError, "items_list"):
            self.make_matrix(items_list=[1, 2]).gen_affinity_matrix()

    def test_unmapped_columns(self):
        matrix = self.make_matrix()
        matrix.gen_affinity_matrix()
        with self.assertRaisesRegex(ValueError, "shape"):
            matrix.map_back_sparse(np.ones((1, 2)), "ratings")

    def test_path_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_matrix(save_path=Path(tmp)).gen_affinity_matrix()
            self.assertTrue((Path(tmp) / "user_dict.npy").is_file())
