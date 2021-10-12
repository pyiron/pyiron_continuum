# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import PyironTestCase
from pyiron_continuum.schroedinger.potentials import SquareWell, Sinusoidal
from pyiron_continuum.mesh import RectMesh
import numpy as np


class _PotentialTest(PyironTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mesh = RectMesh([1, 1], 6)


class TestSquareWell(_PotentialTest):
    def test_input(self):
        potl = SquareWell(width=1/3, depth=2)
        self.assertTrue(np.allclose(
            [
                6*[2],
                6*[2],
                [2, 2, 0, 0, 2, 2],
                [2, 2, 0, 0, 2, 2],
                6 * [2],
                6 * [2],
            ],
            potl(self.mesh)
        ))

    def test_call(self):
        not_a_mesh = np.random.rand(*self.mesh.shape)
        potl = SquareWell()
        with self.assertRaises(AttributeError):
            potl(not_a_mesh)  # Relies on mesh attributes


class TestSinusoidal(_PotentialTest):
    def test_input(self):
        potl = Sinusoidal(n_waves=2, amplitude=2)
        self.assertTrue(np.allclose(
            [
                [0,  0,  0, 0,  0,  0],
                [0,  3, -3, 0,  3, -3],
                [0, -3,  3, 0, -3,  3],
                [0,  0,  0,  0,  0, 0],
                [0,  3, -3, 0,  3, -3],
                [0, -3,  3, 0, -3,  3]
            ],
            potl(self.mesh)
        ))

    def test_call(self):
        not_a_mesh = np.random.rand(*self.mesh.shape)
        potl = Sinusoidal()
        with self.assertRaises(AttributeError):
            potl(not_a_mesh)  # Relies on mesh attributes
