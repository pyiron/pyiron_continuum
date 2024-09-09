# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_continuum.damask.reference.yaml import list_plasticity, list_elasticity


class TestReference(unittest.TestCase):
    def test_content(self):
        plasticity = list_plasticity()
        self.assertIsInstance(plasticity, dict)
        self.assertGreater(len(plasticity), 0)

        elasticity = list_elasticity()
        self.assertIsInstance(elasticity, dict)
        self.assertGreater(len(elasticity), 0)


if __name__ == "__main__":
    unittest.main()
