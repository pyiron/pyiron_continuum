# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_continuum.reference.elastic_constants import get_elastic_constants
import numpy as np


class TestDecorators(unittest.TestCase):
    def test_elastic_constants(self):
        d = get_elastic_constants("Fe")
        self.assertTrue(all(key in d for key in ["C_11", "C_12", "C_44"]))
        for value in d.values():
            self.assertTrue(np.isscalar(value))
            self.assertGreater(value, 0)


if __name__ == "__main__":
    unittest.main()
