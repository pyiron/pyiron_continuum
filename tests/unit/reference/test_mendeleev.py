# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_continuum.reference.mendeleev import get_atom_info
import numpy as np


class TestMendeleev(unittest.TestCase):
    def test_data(self):
        self.assertEqual(get_atom_info("aluminium", "name", "symbol"), "Al")
        self.assertRaises(
            KeyError, get_atom_info, "aluminium", "name", "symbol", 0.99
        )
        self.assertRaises(
            KeyError, get_atom_info, "aluminium", "name", "symbol", 1
        )


if __name__ == "__main__":
    unittest.main()
