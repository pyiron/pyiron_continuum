# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A job class for solving the time-independent Schroedinger equation on a discrete mesh.
"""

from pyiron_base import HasStorage
import numpy as np
from typing import Union, List, Callable, Any
BoundsList = List[Union[float, int, List[Union[float, int]]]]


class RectMesh(HasStorage):
    """
    A helper class for building rectangular meshgrids in n-dimensions.
    Assumes periodic boundary conditions.
    Mesh corresponds to numpy meshgrid with `indexing='ij'` *not* the default `'xy'` indexing. This gives consistency
    in behaviour for meshes in >2 dimensions.

    Example 1-D)

    >>> mesh = RectMesh(2, 2, simplify_1d=True)
    >>> mesh.mesh
    array([0., 1.])
    >>> mesh.steps
    array(1.)

    Example 2-D)
    >>> mesh = RectMesh(bounds=[[0, 1], [2, 5]], divisions=[2, 3])
    >>> mesh.mesh
    array([[[0. , 0. , 0. ],
            [0.5, 0.5, 0.5]],
    <BLANKLINE>
           [[2. , 3. , 4. ],
            [2. , 3. , 4. ]]])
    >>> mesh.steps
    array([0.5, 1. ])

    Note: To get a 1D mesh starting somewhere other than 0, you need an extra set of brackets, i.e.
    `bounds=[[start, end]]` as simply using `[start, end]` will actually give you a 2D array `[[0, start], [0, end]]`!

    Attributes:
        bounds (numpy.ndarray): The start and end point for each dimension.
        divisions (numpy.ndarray): How many sampling points in each dimension.
        mesh (numpy.ndarray): The spatial sampling points.
        steps (numpy.ndarray/float): The step size in each dimension.
        lengths (numpy.ndarray/float): How large the domain is in each dimension.
        simplify_1d (bool): Whether to reduce dimension whenever the first dimension is redundant, e.g. [[1,2]]->[1,2].

    Methods:
        laplacian: Given a callable that takes the mesh as its argument, or a numpy array with the same shape as the
            mesh's real-space dimensions (i.e. `self.shape[1:]`, since the first mesh dimension maps over the dimensions
            themselves) returns the discrete Laplace operator on this mesh applied to that funcation/data.
    """

    def __init__(
            self,
            bounds: Union[float, int, BoundsList, np.ndarray] = 1,
            divisions: Union[int, List[int], np.ndarray] = 1,
            simplify_1d: bool = False
    ):
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
    def bounds(self) -> np.ndarray:
        return self.storage.bounds

    @bounds.setter
    def bounds(self, new_bounds: Union[float, int, BoundsList, np.ndarray]):
        new_bounds, _ = self._clean_input(new_bounds, self.divisions)
        self.storage.bounds = new_bounds
        self._build_mesh()

    @property
    def divisions(self) -> Union[int, np.ndarray]:
        return self.storage.divisions

    @divisions.setter
    def divisions(self, new_divisions: Union[int, List[int], np.ndarray]):
        _, new_divisions = self._clean_input(self.bounds, new_divisions)
        self.storage.divisions = new_divisions
        self._build_mesh()

    @property
    def simplify_1d(self) -> bool:
        return self.storage.simplify_1d

    @simplify_1d.setter
    def simplify_1d(self, simplify: bool):
        self.storage.simplify_1d = simplify

    def _simplify_1d(self, x: np.ndarray) -> Union[int, float, np.ndarray]:
        if len(x) == 1 and self.storage.simplify_1d:
            return np.squeeze(x)
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
    def shape(self) -> tuple:
        return self.mesh.shape

    @property
    def dim(self) -> int:
        return self.shape[0]

    @property
    def lengths(self) -> Union[float, np.ndarray]:
        return self._simplify_1d(self.bounds.ptp(axis=-1))

    @property
    def volume(self):
        return self.lengths.prod()

    def _build_mesh(self) -> None:
        linspaces = []
        steps = []
        for bound, ndiv in zip(self.storage.bounds, self.storage.divisions):
            space, step = np.linspace(bound[0], bound[1], num=ndiv, endpoint=False, retstep=True)
            linspaces.append(space)
            steps.append(step)

        mesh = np.meshgrid(*linspaces, indexing='ij')
        self.storage.steps = np.array(steps)
        self.storage.mesh = np.array(mesh)

    def _clean_input(
            self,
            bounds: Union[float, int, BoundsList, np.ndarray],
            divisions: Union[int, List[int], np.ndarray]
    ) -> (np.ndarray, np.ndarray):
        if not hasattr(bounds, '__len__'):
            bounds = [[0, bounds]]
        bounds = np.array(bounds)  # Enforce array to guarantee `shape`

        if len(bounds.shape) == 1:
            bounds = np.array([[0, b] for b in bounds])
        elif len(bounds.shape) > 2 or bounds.shape[-1] > 2:
            raise ValueError(f'Bounds must be of the shape (n,) or (n, 2), but got {bounds.shape}')

        if np.any(np.isclose(bounds.ptp(axis=-1), 0)):
            raise ValueError(f'Bounds must be finite length in all dimensions, but found lengths {bounds.ptp(axis=-1)}')

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
    def _is_int(val: Any) -> bool:
        return np.issubdtype(type(val), np.integer)

    def laplacian(self, fnc: Union[Callable, np.ndarray]) -> np.array:
        """
        Discrete Laplacian operator applied to a given function or scalar field.

        Args:
            fnc (function/numpy.ndarray): A function taking the `mesh.mesh` value and returning a scalar field, or the
                scalar field

        Returns:
            (numpy.ndarray): The scalar field Laplacian of the function at each point on the grid.
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
