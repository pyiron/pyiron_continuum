# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_continuum.damask import factory


class TestDamaskFactory(unittest.TestCase):
    def test_generate_load_step(self):
        key, value = factory.generate_loading_tensor(default="P")
        self.assertTrue(key[0, 0] == "P")
        key, value = factory.generate_loading_tensor()
        value[0, 0] = 0
        key[0, 0] = "P"
        key[1, 1] = "dot_P"
        key[2, 2] = "dot_F"
        d = factory.loading_tensor_to_dict(key, value)
        data = factory.generate_load_step(N=40, t=10, **d)
        self.assertTrue("P" in data["boundary_conditions"]["mechanical"])
        for tag in ["f_out", "r", "f_restart", "estimate_rate"]:
            d = factory.loading_tensor_to_dict(key, value)
            d.update({tag: 1})
            data = factory.generate_load_step(N=40, t=10, **d)
            if tag != "r":
                self.assertTrue(tag in data)
            else:
                self.assertTrue(tag in data["discretization"])

    def test_generate_grid_from_voronoi_tesselation(self):
        grid = factory.generate_grid_from_voronoi_tessellation(16, 8, 1e-5)
        self.assertAlmostEquals(grid.size.max(), 1.0e-5)
        self.assertAlmostEquals(grid.size.min(), 1.0e-5)



if __name__ == "__main__":
    unittest.main()
