# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for solving the time-independent Schroedinger equation on a discrete mesh.
"""

from pyiron_base import HasStorage
import numpy as np


class RectMesh(HasStorage):
    """
    A helper class for building rectangular meshgrids in 1-, 2-, or 3-D

    Example 1-D)

    >>> mesh = RectMesh(2, 2)
    >>> mesh.mesh
    array([0., 1.])
    >>> mesh.steps
    1.0

    Example 2-D)
    >>> mesh = RectMesh(bounds=[[0, 1], [2, 5]], divisions=[2, 3])
    >>> mesh.mesh
    array([[[0. , 0.5],
            [0. , 0.5],
            [0. , 0.5]],
    <BLANKLINE>
           [[2. , 2. ],
            [3. , 3. ],
            [4. , 4. ]]])
    >>> mesh.steps
    array([0.5, 1. ])

    Attributes:
        bounds (numpy.ndarray): The start and end point for each dimension.
        divisions (numpy.ndarray): How many sampling points in each dimension.
        mesh (numpy.ndarray): The spatial sampling points.
        steps (numpy.ndarray): The step size in each dimension.
    """

    def __init__(self, bounds, divisions, simplify_1d=True):
        super().__init__()
        bounds, divisions = self._clean_input(bounds, divisions)
        self.storage.bounds = bounds
        self.storage.divisions = divisions
        self.storage.simplify_1d = simplify_1d
        self._build_mesh()

    @property
    def bounds(self):
        return self.storage.bounds

    @bounds.setter
    def bounds(self, new_bounds):
        new_bounds, _ = self._clean_input(new_bounds, self.divisions)
        self.storage.bounds = new_bounds
        self._build_mesh()

    @property
    def divisions(self):
        return self.storage.divisions

    @divisions.setter
    def divisions(self, new_divisions):
        _, new_divisions = self._clean_input(self.bounds, new_divisions)
        self.storage.divisions = new_divisions
        self._build_mesh()

    @property
    def simplify_1d(self):
        return self.storage.simplify_1d

    @simplify_1d.setter
    def simplify_1d(self, simplify: bool):
        self.storage.simplify_1d = simplify

    def _simplify_1d(self, x):
        if len(x) == 1 and self.storage.simplify_1d:
            return x[0]   # 1D
        else:
            return x

    @property
    def mesh(self) -> np.ndarray:
        return self._simplify_1d(self.storage.mesh)

    @property
    def steps(self) -> np.ndarray:
        return self._simplify_1d(self.storage.steps)

    def _build_mesh(self):
        linspaces = []
        steps = []
        for bound, ndiv in zip(self.storage.bounds, self.storage.divisions):
            space, step = np.linspace(bound[0], bound[1], num=ndiv, endpoint=False, retstep=True)
            linspaces.append(space)
            steps.append(step)

        mesh = np.meshgrid(*linspaces)
        self.storage.steps = np.array(steps)
        self.storage.mesh = np.array(mesh)

    def _clean_input(self, bounds, divisions):
        if not hasattr(bounds, '__len__'):
            bounds = [[0, bounds]]
        bounds = np.array(bounds)  # Enforce array to guarantee `shape`

        if len(bounds.shape) == 1:
            bounds = np.array([[0, b] for b in bounds])

        if np.any(bounds.shape > np.array([3, 2])):
            raise ValueError(f'Bounds can be shape (3,2) at the largest, but got {bounds.shape}')

        if hasattr(divisions, '__len__'):
            if len(divisions) != len(bounds):
                raise ValueError(
                    f'Divisions must be a single value or have the same length as bounds but got {len(divisions)} and '
                    f'{len(bounds)}'
                )
            elif np.any([not self._is_int(div) for div in divisions]):
                raise TypeError(f'Divisions must all be int-like, but got {divisions}')
        elif self._is_int(divisions):
            divisions = len(bounds) * [divisions]
        else:
            raise TypeError(
                f'Expected divisions to be int-like or a list-like objects of ints the same length as bounds, but got '
                f'{divisions}'
            )
        return bounds, np.array(divisions)

    @staticmethod
    def _is_int(val):
        return np.issubdtype(type(val), np.integer)

    def laplacian(self, fnc):
        """
        Discrete Laplacian operator applied to a given function assuming periodic boundary conditions.

        Args:
            fnc (function): A function taking the `mesh.mesh` value.

        Returns:
            (numpy.ndarray): The Laplacian of the function at each point on the grid.
        """
        val = fnc(self.mesh)
        res = np.zeros(val.shape)
        for ax, ds in enumerate(self.steps):
            res += (np.roll(val, 1, axis=ax) + np.roll(val, -1, axis=ax) - 2 * val) / ds ** 2
        return res
