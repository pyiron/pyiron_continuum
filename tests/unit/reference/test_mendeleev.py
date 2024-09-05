# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_continuum.reference.mendeleev import get_atom_info


class TestMendeleev(unittest.TestCase):
    def test_data(self):
        self.assertEqual(get_atom_info(name="aluminium")["symbol"], "Al")
        with self.assertRaises(KeyError):
            print(get_atom_info(name="my dog chased a cat"))
        self.assertRaises(ValueError, get_atom_info)


if __name__ == "__main__":
    unittest.main()
