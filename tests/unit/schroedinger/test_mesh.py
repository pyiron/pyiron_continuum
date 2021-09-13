# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pyiron_base._tests import PyironTestCase
from pyiron_continuum.schroedinger.mesh import RectMesh
import numpy as np
import pyiron_continuum.schroedinger.mesh as mesh_mod


class TestRectMesh(PyironTestCase):
    def setUp(self) -> None:
        super().setUp()

    @property
    def docstring_module(self):
        return mesh_mod

    def test_input(self):
        L = 2*np.pi
        n = 2
        mesh = RectMesh(L, n)
        self.assertTrue(np.allclose(mesh.bounds, [[0, L]]),
                        msg='Expected float to be converted to (1,2) array.')
        self.assertTrue(np.all(mesh.divisions == [n]),
                        msg='Expected int to be converted to (1,) array.')

        mesh = RectMesh([L, L], n)
        self.assertTrue(np.allclose(mesh.bounds, [[0, L], [0, L]]),
                        msg='Expected 1D bounds to be interpreted as endpoints of 2D bounds.')
        self.assertTrue(np.all(mesh.divisions == [n, n]),
                        msg='Expected divisions to be extended to match bounds.')

        mesh = RectMesh([[0, L], [L / 2, L]], [n, 2 * n])
        self.assertTrue(np.allclose(mesh.bounds, [[0, L], [L/2, L]]),
                        msg='Expected float to be converted to (1,2) array.')
        self.assertTrue(np.all(mesh.divisions == [n, 2*n]),
                        msg='Expected divisions to be preserved.')

        self.assertRaises(ValueError, RectMesh, [[0, 1, 2]], 1)  # Bounds can't exceed shape (3, 2)
        self.assertRaises(ValueError, RectMesh, [1, 1, 1, 1], 1)  # Bounds can't exceed shape (3, 2)
        self.assertRaises(ValueError, RectMesh, 1, [1, 1])  # Divisions must be a single value or match bounds
        self.assertRaises(TypeError, RectMesh, 1, np.pi)  # Only int-like divisions
        self.assertRaises(TypeError, RectMesh, 1, [[1]])  # Or lists of ints, nothing else like lists of lists

    def test_construction(self):
        L = 2 * np.pi
        n = 2
        mesh = RectMesh(L, n)
        self.assertTrue(np.allclose(mesh.mesh, [0, L/2]), msg='1D should get simplified')
        self.assertAlmostEqual(mesh.steps, L/2, msg='1D should get simplified')
        mesh.simplify_1d = False
        self.assertTrue(np.allclose(mesh.steps, [L/2]), msg='1D should stay list-like')

        mesh = RectMesh([L, 2 * L], n)
        self.assertTrue(
            np.allclose(
                mesh.mesh,
                [
                    [
                        [0, L / 2],
                        [0, L / 2],
                    ],
                    [
                        [0, 0],
                        [L, L],
                    ]
                ]
            )
        )
        self.assertTrue(np.allclose(mesh.steps, [L / 2, L]))

    def test_update(self):
        L = 2 * np.pi
        n = 2
        mesh = RectMesh([L, L], n)
        init_mesh = np.array(mesh.mesh)
        mesh.bounds = [2 * L, 2 * L]
        self.assertTrue(np.allclose(mesh.mesh, 2 * init_mesh), msg='Should have doubled extent')
        mesh.divisions = 2 * n
        self.assertEqual(2 * n, len(mesh.mesh[0]), msg='Should have doubled sampling density')

    def test_laplacian(self):
        """
        Check that the numeric laplacian matches an analytic reference better for denser meshes, even when mesh spacings
        differ across dimensions.
        """
        def fnc(mesh):
            """Product of sines. Analytic Laplacian is negative product of sines multiplied by number of dimensions."""
            return np.prod([np.sin(m) for m in mesh.mesh], axis=0)

        for dims in [1, 2, 3]:
            convergence = []
            for divs in [10, 50]:
                mesh = RectMesh(dims * [2 * np.pi], [divs + d for d in range(dims)])
                analytic = -dims * fnc(mesh)
                numeric = mesh.laplacian(fnc)
                convergence.append(np.linalg.norm(analytic - numeric))
            self.assertLess(convergence[1], convergence[0], msg='Expected a better solution with a denser mesh.')

    def test_length(self):
        mesh = RectMesh([[0, 1], [1, 3]], 2)
        self.assertAlmostEqual(1, mesh.lengths[0], msg='Real-space length in x-direction should be 1')
        self.assertAlmostEqual(2, mesh.lengths[1], msg='Real-space length in y-direction should be 3-1=2')
