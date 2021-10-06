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


def callable_to_array(method):
    """If the first argument of the method is callable, replaces it with the callable evaluated on self."""
    def wrapper(self, fnc, **kwargs):
        if callable(fnc):
            return method(self, fnc(self), **kwargs)
        else:
            return method(self, fnc, **kwargs)
    return wrapper


def takes_scalar_field(method):
    """Makes sure the first argument has the right shape to be a scalar field on the mesh."""
    def wrapper(self, scalar_field, **kwargs):
        scalar_field = np.array(scalar_field)
        if np.all(scalar_field.shape == self.divisions):
            return method(self, scalar_field, **kwargs)
        else:
            raise TypeError(f'Argument for {method.__name__} not recognized: should be a scalar field, or function '
                            f'taking the mesh and returning a scalar field.')
    return wrapper


def takes_vector_field(method):
    """Makes sure the first argument has the right shape to be a vector field on the mesh."""
    def wrapper(self, vector_field, **kwargs):
        vector_field = np.array(vector_field)
        if np.all(vector_field.shape == self.shape):
            return method(self, vector_field, **kwargs)
        else:
            raise TypeError(f'Argument for {method.__name__} not recognized: should be a vector field, or function '
                            f'taking the mesh and returning a vector field.')
    return wrapper


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

    def __len__(self):
        return self.divisions.prod()

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

    @callable_to_array
    @takes_scalar_field
    def laplacian(self, scalar_field: Union[Callable, np.ndarray]) -> np.array:
        """
        Discrete Laplacian operator applied to a given function or scalar field.

        Args:
            fnc (function/numpy.ndarray): A function taking the `mesh.mesh` value and returning a scalar field, or the
                scalar field

        Returns:
            (numpy.ndarray): The scalar field Laplacian of the function at each point on the grid.
        """
        res = np.zeros(self.divisions)
        for ax, ds in enumerate(self.steps):
            res += (np.roll(scalar_field, 1, axis=ax) + np.roll(scalar_field, -1, axis=ax) - 2 * scalar_field) / ds ** 2
        return res

    @callable_to_array
    @takes_scalar_field
    def grad(self, scalar_field: Union[Callable, np.ndarray], order: int = 2) -> np.array:
        """
        Gradient operator applied to a given function or scalar field.

        Args:
            scalar_field (function/numpy.ndarray): A function taking the `mesh.mesh` value and returning a scalar field,
                or the scalar field as an array.
            order (int): The order of approximation, 1 uses two points, 2 uses four points. (Default is 2.)

        Returns:
            (numpy.ndarray): The vector field gradient of the function at each point on the grid.
        """
        return self.derivative(scalar_field, order=1, accuracy=order)

    @callable_to_array
    @takes_vector_field
    def div(self, vector_field: Union[Callable, np.ndarray], order: int = 2) -> np.array:
        res = np.zeros(self.divisions)
        for ax in np.arange(self.dim):
            res += self.grad(vector_field[ax], order=order)[ax]
        return res

    @callable_to_array
    @takes_vector_field
    def curl(self, vector_field: Union[Callable, np.ndarray], order: int = 2) -> np.array:
        if self.dim != 3:
            raise NotImplementedError("I'm no mathematician, so curl is only coded for the traditional 3d space.")
        grads = np.array([self.grad(vf, order=order) for vf in vector_field])
        pos = np.array([grads[(2 + i) % self.dim][(1 + i) % self.dim] for i in range(self.dim)])
        neg = np.array([grads[(1 + i) % self.dim][(2 + i) % self.dim] for i in range(self.dim)])
        return pos - neg

    @callable_to_array
    @takes_scalar_field
    def derivative(self, scalar_field: Union[Callable, np.ndarray], order=1, accuracy=2):
        """
        Numeric differential for a uniform grid using the central difference method.

        Args:
            scalar_field (function/numpy.ndarray): A function taking the `mesh.mesh` value and returning a scalar field,
                or the scalar field as an array.
            order (int): The derivative to take. (Default is 1, take first derivative.)
            accuracy (int): The accuracy of the method in O(grid spacing). (Default is 2, O(h^2) accuracy).

        Returns:
            (numpy.ndarray): The vector field derivative of the function at each point on the grid in each dimension.

        Raises:
            (KeyError): If the requested order or accuracy cannot be found.
        """
        try:
            coefficients = self.central_difference_table[order][accuracy]
        except KeyError:
            raise KeyError(f'The requested order {order} and accuracy {accuracy} could not be found. Please look at '
                           f'the `central_difference_table` attribute to see all available choices.')

        res = np.zeros(self.shape)
        for ax, h in enumerate(self.steps):
            for n, c in coefficients.items():
                res[ax] += c * np.roll(scalar_field, n, axis=ax)
            res[ax] /= h**order
        return res


    @property
    def central_difference_table(self):
        """
        Coefficients for numeric differentials sorted by order of differential and accuracy of method:
        - order of differential
            - accuracy of method
                - roll: coefficient

        Cf. [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference_coefficient)

        TODO: Fill out the rest of the values
        """
        return {
            1: {
                2: {1: -1/2, -1: 1/2},
                4: {2: 1/12, 1: -2/3, -1: 2/3, -2: -1/12},
                # 6: {},
                # 8: {},
            },
            2: {
                2: {1: 1, 0: -2, -1: 1},
                # 4: {},
                # 6: {},
                # 8: {},
            },
        #     3: {
        #         2: {},
        #         4: {},
        #         6: {},
        #     },
        #     4: {
        #         2: {},
        #         4: {},
        #         6: {},
        #     },
        #     5: {
        #         2: {},
        #         4: {},
        #         6: {},
        #     },
        #     6: {
        #         2: {},
        #         4: {},
        #         6: {},
        #     },
        }
