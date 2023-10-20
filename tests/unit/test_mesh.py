# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest
from pyiron_base._tests import PyironTestCase
from pyiron_continuum.mesh import (
    RectMesh,
    callable_to_array,
    takes_scalar_field,
    takes_vector_field,
    has_default_accuracy
)
import numpy as np
import pyiron_continuum.mesh as mesh_mod


class TestDecorators(PyironTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.mesh = RectMesh([1, 2, 3], [30, 20, 10])

    @staticmethod
    def give_vector(mesh):
        return np.ones(mesh.shape)

    @staticmethod
    def give_scalar(mesh):
        return np.ones(mesh.divisions)

    def test_callable_to_array(self):
        scalar_field = self.give_scalar(self.mesh)

        @callable_to_array
        def method(mesh, callable_or_array, some_kwarg=1):
            return callable_or_array + some_kwarg

        self.assertTrue(np.allclose(scalar_field + 1, method(self.mesh, self.give_scalar)), msg="Accept functions")
        self.assertTrue(np.allclose(scalar_field + 1, method(self.mesh, scalar_field)), msg="Accept arrays")
        self.assertTrue(np.allclose(scalar_field + 2, method(self.mesh, self.give_scalar, some_kwarg=2)),
                        msg="Pass kwargs")

    def test_takes_scalar_field(self):
        scalar_field = self.give_scalar(self.mesh)

        @takes_scalar_field
        def method(mesh, scalar_field, some_kwarg=1):
            return some_kwarg

        self.assertEqual(1, method(self.mesh, scalar_field), msg="Accept arrays")
        self.assertEqual(2, method(self.mesh, scalar_field, some_kwarg=2), msg="Pass kwargs")
        self.assertEqual(1, method(self.mesh, scalar_field.tolist()), msg="Should work with listlike stuff too")
        self.assertRaises(TypeError, method, self.mesh, np.ones(2))  # Reject the wrong shape
        self.assertRaises(ValueError, method, self.mesh, "not even numeric")  # Duh

    def test_takes_vector_field(self):
        vector_field = self.give_vector(self.mesh)

        @takes_vector_field
        def method(mesh, vector_field, some_kwarg=1):
            return some_kwarg

        self.assertEqual(1, method(self.mesh, vector_field), msg="Accept arrays")
        self.assertEqual(2, method(self.mesh, vector_field, some_kwarg=2), msg="Pass kwargs")
        self.assertEqual(1, method(self.mesh, vector_field.tolist()), msg="Should work with listlike stuff too")
        self.assertRaises(TypeError, method, self.mesh, np.ones(2))  # Reject the wrong shape
        self.assertRaises(TypeError, method, self.mesh, "not even numeric")  # Duh

    def test_has_default_accuracy(self):
        some_field = self.give_vector(self.mesh)

        @has_default_accuracy
        def method(mesh, field, accuracy=None, some_kwarg=1):
            return accuracy + some_kwarg

        mesh = RectMesh(1, 1, accuracy=2)

        self.assertEqual(3, method(mesh, some_field), 'Use mesh accuracy')
        self.assertEqual(0, method(mesh, some_field, accuracy=4, some_kwarg=-4), 'Use passed accuracy')
        self.assertRaises(ValueError, method, mesh, some_field, accuracy=1)  # Even accuracy only
        self.assertRaises(ValueError, method, mesh, some_field, accuracy=0)  # Positive accuracy only

        @has_default_accuracy
        def method(mesh, field, accuracy_not_a_kwarg=42):
            return None

        self.assertRaises(TypeError, method, mesh, some_field)  # Methods need to define accuracy


class TestRectMesh(PyironTestCase):

    @staticmethod
    def scalar_sines(mesh):
        L = mesh.lengths
        omega = (2 * np.pi / L).reshape(len(L), *[1] * mesh.dim)
        return np.prod(np.sin(omega * mesh.mesh), axis=0)

    def vector_sines(self, mesh):
        scalar = self.scalar_sines(mesh)
        return np.array(mesh.dim * [scalar])

    @property
    def docstring_module(self):
        return mesh_mod

    def test_input(self):
        L = np.pi
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
        self.assertTrue(np.allclose(mesh.bounds, [[0, L], [L / 2, L]]),
                        msg='Expected float to be converted to (1,2) array.')
        self.assertTrue(np.all(mesh.divisions == [n, 2 * n]),
                        msg='Expected divisions to be preserved.')

        bounds = np.array([1, 2, 3, 4])
        self.assertAlmostEqual(
            bounds.prod(),
            RectMesh(bounds=bounds).volume,
            msg="Four dimensions should be ok, and hyper-volume should be a product of side lengths"
        )

        self.assertRaises(ValueError, RectMesh, [[0, 1, 2]], 1)  # Bounds can't exceed shape (n, 2)
        self.assertRaises(ValueError, RectMesh, [[1, 1 + 1e-12]])  # Bounds must enclose a space noticeably > 0
        self.assertRaises(ValueError, RectMesh, 1, [1, 1])  # Divisions must be a single value or match bounds
        self.assertRaises(TypeError, RectMesh, 1, np.pi)  # Only int-like divisions
        self.assertRaises(TypeError, RectMesh, 1, [[1]])  # Or lists of ints, but nothing else like lists of lists

    def test_construction(self):
        L = np.pi
        n = 2
        mesh = RectMesh(L, n)
        self.assertTrue(np.allclose(mesh.mesh, [0, L / 2]), msg='1D should get simplified')
        self.assertAlmostEqual(mesh.steps, L / 2, msg='1D should get simplified')
        mesh.simplify_1d = False
        self.assertTrue(np.allclose(mesh.steps, [L / 2]), msg='1D should stay list-like')

        mesh = RectMesh([L, 2 * L], n)
        self.assertTrue(
            np.allclose(
                mesh.mesh,
                [
                    [
                        [0, 0],
                        [L / 2, L / 2],
                    ],
                    [
                        [0, L],
                        [0, L],
                    ]
                ]
            )
        )
        self.assertTrue(np.allclose(mesh.steps, [L / 2, L]))

        with self.assertRaises(ValueError):
            RectMesh([[1, 1+1E-12]])  # Mesh needs finite length

    def test_update(self):
        L = np.pi
        n = 2
        mesh = RectMesh([L, L], n)
        init_mesh = np.array(mesh.mesh)
        mesh.bounds = [2 * L, 2 * L]
        self.assertTrue(np.allclose(mesh.mesh, 2 * init_mesh), msg='Should have doubled extent')
        mesh.divisions = 2 * n
        self.assertEqual(2 * n, len(mesh.mesh[0]), msg='Should have doubled sampling density')

    def test_length(self):
        mesh = RectMesh([[0, 1], [1, 3]], 2)
        self.assertAlmostEqual(1, mesh.lengths[0], msg='Real-space length in x-direction should be 1')
        self.assertAlmostEqual(2, mesh.lengths[1], msg='Real-space length in y-direction should be 3-1=2')

    def test_derivative(self):
        L = np.pi
        omega = 2 * np.pi / L
        mesh = RectMesh(L, 100)
        x = mesh.mesh[0]

        def solution(order):
            """derivatives of sin(omega * x)"""
            fnc = np.cos(omega * x) if order % 2 == 1 else np.sin(omega * x)
            sign = -1 if order % 4 in [2, 3] else 1
            return sign * omega**order * fnc

        for order in [1, 2, 3, 4]:
            errors = [
                np.linalg.norm(solution(order) - mesh.derivative(self.scalar_sines, order=order, accuracy=accuracy))
                for accuracy in [2, 4, 6]
            ]
            self.assertTrue(np.all(np.diff(errors) < 0), msg="Increasing accuracy should give decreasing error.")

        self.assertRaises(ValueError, mesh.derivative, mesh.mesh[0], order=1, accuracy=3)  # No odd accuracies

    def test_grad(self):
        L = np.pi
        omega = 2 * np.pi / L
        mesh = RectMesh([L, L, L], [100, 200, 300])

        x, y, z = mesh.mesh
        solution = np.array([
            omega * np.cos(x * omega) * np.sin(y * omega) * np.sin(z * omega),
            omega * np.sin(x * omega) * np.cos(y * omega) * np.sin(z * omega),
            omega * np.sin(x * omega) * np.sin(y * omega) * np.cos(z * omega)
        ])

        grad1 = mesh.grad(self.scalar_sines, accuracy=2)  # Can take function
        err1 = np.linalg.norm(grad1 - solution) / len(mesh)
        grad2 = mesh.grad(self.scalar_sines(mesh), accuracy=4)  # Or a numpy array directly
        err2 = np.linalg.norm(grad2 - solution) / len(mesh)
        self.assertLess(err1, 0.001, msg="Should a pretty good approximation")
        self.assertAlmostEquals(0, err2, msg="Should be a very good approximation")
        self.assertLess(err2, err1, msg="Second order approximation should be better")

        self.assertRaises(TypeError, mesh.grad, np.random.rand(1, 2, 3))  # Values aren't a nice scalar field on mesh!

    def test_div(self):
        L = np.pi
        omega = 2 * np.pi / L
        mesh = RectMesh([L, L, L, L], 50)  # Unlike curl, div is not restricted to 3d

        sinx, siny, sinz, sinw = np.sin(omega * mesh.mesh)
        cosx, cosy, cosz, cosw = np.cos(omega * mesh.mesh)
        solution = omega * np.array([
            cosx * siny * sinz * sinw
            + sinx * cosy * sinz * sinw
            + sinx * siny * cosz * sinw
            + sinx * siny * sinz * cosw
        ])

        self.assertTrue(np.allclose(solution, mesh.div(self.vector_sines)), msg="Differentiating sines is not hard")
        self.assertTrue(np.allclose(solution, mesh.div(self.vector_sines(mesh))), msg="Should work with arrays too")

    def test_laplacian(self):
        """
        Check that the numeric laplacian matches an analytic reference better for denser meshes, even when mesh spacings
        differ across dimensions.
        """

        def solution(mesh):
            """
            Analytic Laplacian for product of sines is negative product of sines multiplied by number of dimensions.
            """
            return -self.scalar_sines(mesh) * mesh.dim

        for dims in [1, 2, 3]:
            convergence = []
            for divs in [10, 50]:
                mesh = RectMesh(dims * [2 * np.pi], [divs + d for d in range(dims)])
                analytic = solution(mesh)
                numeric = mesh.laplacian(self.scalar_sines)
                convergence.append(np.linalg.norm(analytic - numeric))
            self.assertLess(convergence[1], convergence[0], msg='Expected a better solution with a denser mesh.')

    def test_curl(self):
        L = 1
        omega = 2 * np.pi / L
        mesh = RectMesh([L, L, L], 100)

        x, y, z = mesh.mesh
        solution = np.array([
            omega * np.sin(omega * x) * (np.sin(omega * z) * np.cos(omega * y) - np.sin(omega * y) * np.cos(omega * z)),
            omega * np.sin(omega * y) * (np.sin(omega * x) * np.cos(omega * z) - np.sin(omega * z) * np.cos(omega * x)),
            omega * np.sin(omega * z) * (np.sin(omega * y) * np.cos(omega * x) - np.sin(omega * x) * np.cos(omega * y))
        ])

        self.assertTrue(np.allclose(solution, mesh.curl(self.vector_sines)), msg="Should work with callable")
        self.assertTrue(np.allclose(solution, mesh.curl(self.vector_sines(mesh))), msg="Should work with array")

        mesh2 = RectMesh([1, 1], 2)
        self.assertRaises(NotImplementedError, mesh2.curl, mesh2.mesh)  # Should not work for dimensions other than 3

if __name__ == "__main__":
    unittest.main()
