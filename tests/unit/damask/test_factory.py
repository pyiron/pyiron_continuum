# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_continuum.damask import factory


class TestDamaskFactory(unittest.TestCase):
    def test_generate_load_step(self):
        key, value = factory.generate_loading_tensor()
        value[0, 0] = 0
        key[0, 0] = "P"
        data = factory.generate_load_step(
            boundary_conditions=(key, value), N=40, t=10
        )
        self.assertTrue("P" in data["boundary_conditions"]["mechanical"])
        for tag in ["f_out", "r", "f_restart", "estimate_rate"]:
            data = factory.generate_load_step(
                boundary_conditions=(key, value), N=40, t=10, **{tag: 1}
            )
            if tag != "r":
                self.assertTrue(tag in data)
            else:
                self.assertTrue(tag in data["discretization"])


if __name__ == "__main__":
    unittest.main()
