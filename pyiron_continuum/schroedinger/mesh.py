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

    >>> mesh = RectMesh(2, 2, simplify_1d=True)
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

    Methods:
        laplacian: Given a callable that takes the mesh as its argument, or a numpy array with the same shape as the
            mesh's real-space dimensions (i.e. `self.shape[1:]`, since the first mesh dimension maps over the dimensions
            themselves) returns the discrete Laplace operator on this mesh applied to that funcation/data.
    """

    def __init__(self, bounds, divisions, simplify_1d=False):
        """
        Instantiate a rectangular mesh.

        Args:
            bounds (float/list/numpy.ndarray): The upper and lower bounds for each dimension of the mesh. A single
                value, L, creates a mesh on [0, L]. A list/array with shape (n<=3,) creates an n-dimensional mesh with
                on [0, L_i] for each of the L_i values given. A two-dimensional list/array should have shape (n<=3,2)
                and gives lower and upper bounds for each dimension, i.e. `[[0.5, 1]]` makes a 1D mesh on [0.5, 1].
            divisions (int/list/numpy.ndarray): How many grid divisions to use in each dimension. An integer will be
                mapped up to give that number of divisions on all dimensions provided in `bounds`, otherwise the
                dimensionality between the two arguments must agree.
            simplify_1d (bool): Whether to simplify the output for 1D meshes so they have shape (n,) instead of (1, n).
                (Default is False.)
        """
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
        """Spacing between each mesh point."""
        return self._simplify_1d(self.storage.steps)

    @property
    def shape(self):
        return self.mesh.shape

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
            fnc (function/numpy.ndarray): A function taking the `mesh.mesh` value or the function already evaluated.

        Returns:
            (numpy.ndarray): The Laplacian of the function at each point on the grid.
        """
        if callable(fnc):
            val = fnc(self)
        elif np.all(fnc.shape == self.shape[1:]):
            val = fnc
        else:
            raise TypeError('Argument for laplacian not recognized')
        res = np.zeros(val.shape)
        for ax, ds in enumerate(self.steps):
            res += (np.roll(val, 1, axis=ax) + np.roll(val, -1, axis=ax) - 2 * val) / ds ** 2
        return res
