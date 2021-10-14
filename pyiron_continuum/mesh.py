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


def has_default_accuracy(method):
    """Replaces the `accuracy` argument with the instance attribute of the same name if `accuracy` is None."""
    def wrapper(self, fnc, accuracy=None, **kwargs):
        accuracy = self.accuracy if accuracy is None else accuracy
        if accuracy % 2 != 0 or accuracy < 2:
            raise ValueError(f'Expected an even, positive accuracy but got {accuracy}')
        return method(self, fnc, accuracy=accuracy, **kwargs)
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
    **Assumes periodic boundary conditions.**
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
        divisions (numpy.ndarray): How many sampling points in each dimension, i.e. the shape of a scalar field.
        dim (int): The dimensionality of the field.
        shape (tuple): The shape of the mesh, i.e. the shape of a vector field.
        mesh (numpy.ndarray): The spatial sampling points.
        steps (numpy.ndarray/float): The step size in each dimension.
        lengths (numpy.ndarray/float): How large the domain is in each dimension.
        volume (float): The product of the lengths in all dimensions.
        accuracy (int): An even value given the number of stencil points to use in each dimension for calculating
            derivatives. (Default is 4.)
        simplify_1d (bool): Whether to reduce dimension whenever the first dimension is redundant, e.g. [[1,2]]->[1,2].

    Methods:
        derivative: Calculate the nth order derivative of a scalar field to get a vector field.
        grad: Calculate the first derivative of a scalar field to get a vector field.
        div: Calculate the divergence of a vector field to get a scalar field.
        laplacian: Calculate the Laplacian of a scalar field to get a scalar field.
        curl: Calculate the curl of a vector field to get a vector field. (Only for 3d!)

    Note: All the mathematical operations can take either a numpy array of the correct dimension *or* a callable that
        takes the mesh itself as an argument and returns a numpy array of the correct dimension, where 'correct' is
        refering to scalar field (mesh divisions) or vector field (mesh shape).

    Warning: Operations over the actual mesh points are all nicely vectorized, but this is by no means highly optimized
        for numeric efficiency! If you want to do some really heavy lifting, this is probably the wrong tool.

    TODO: Include aperiodic boundary conditions, e.g. padding. Probably by modifying the decorators.
    """

    def __init__(
            self,
            bounds: Union[float, int, BoundsList, np.ndarray] = 1,
            divisions: Union[int, List[int], np.ndarray] = 1,
            accuracy: int = 4,
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
            accuracy (int): An even value given the number of stencil points to use in each dimension for calculating
                derivatives. (Default is 4.)
            simplify_1d (bool): Whether to simplify the output for 1D meshes so they have shape (n,) instead of (1, n).
                (Default is False.)
        """
        super().__init__()
        bounds, divisions = self._clean_input(bounds, divisions)
        self.storage.bounds = bounds
        self.storage.divisions = divisions
        self.storage.accuracy = accuracy
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
    def accuracy(self) -> int:
        """
        The number of points to use in the stencil for central difference methods. Corresponds to O(h^accuracy)
        precision in the derivative operator, where h is the mesh spacing.
        """
        return self.storage.accuracy

    @accuracy.setter
    def accuracy(self, n: int):
        if n % 2 != 0:
            raise ValueError(f'Expected an even integer but got {2}')
        self.storage.accuracy = int(n)

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
        """Dimension of the box, i.e. the zeroth entry of the shape."""
        return self.shape[0]

    @property
    def lengths(self) -> Union[float, np.ndarray]:
        """Edge lengths for each side of the box."""
        return self._simplify_1d(self.bounds.ptp(axis=-1))

    @property
    def volume(self):
        """Volume encompassed by all the dimensions, i.e. product of the lengths."""
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

    @staticmethod
    def _get_central_difference_coefficients(m, n):
        """
        Coefficients for central finite difference numeric differentiation.

        Args:
            m (int): Order of differential.
            n (int): Accuracy of method, i.e. precision as a power of grid spacing.

        Returns:
            (numpy.ndarray): Coefficients for numeric differentials sorted by order of differential and accuracy of
                method.
        """
        if n % 2 != 0:
            raise ValueError('`n` must be an even number')
        p = int(0.5 * (m + 1)) - 1 + int(0.5 * n)
        b = np.zeros(2 * p + 1)
        b[m] = np.prod(np.arange(m) + 1)
        return np.linalg.solve(np.arange(-p, p + 1) ** np.arange(0, 2 * p + 1)[:, None], b)

    # OPERATIONS:
    @callable_to_array
    @takes_scalar_field
    def _numpy_gradient(self, scalar_field: Union[Callable, np.ndarray], axis=None, edge_order=1):
        return np.gradient(scalar_field, *self.steps, axis=axis, edge_order=edge_order)

    @callable_to_array
    @has_default_accuracy
    @takes_scalar_field
    def derivative(self, scalar_field: Union[Callable, np.ndarray], order: int = 1, accuracy: int = None):
        """
        Numeric differential for a uniform grid using the central difference method.

        Args:
            scalar_field (function/numpy.ndarray): A function taking this `RectMesh` object and returning a scalar
                field, or the scalar field as an array.
            order (int): The derivative to take. (Default is 1, take first derivative.)
            accuracy (int): The accuracy of the method in O(grid spacing). (Default is None, which falls back on the
                class attribute of the same name.)

        Returns:
            (numpy.ndarray): The vector field derivative of the scalar input at each point on the grid in each
                dimension. E.g. for a scalar field with shape `(nx, ny, nz)`, returns shape `(3, nx, ny, nz)`.

        Raises:
            (KeyError): If the requested order or accuracy cannot be found.
        """
        coefficients = self._get_central_difference_coefficients(order, accuracy)
        max_roll = (len(coefficients) - 1) / 2
        rolls = np.flip(np.arange(-max_roll, max_roll + 1, dtype=int))

        res = np.zeros(self.shape)
        for ax, h in enumerate(self.steps):
            for n, c in enumerate(coefficients):
                if np.isclose(c, 0):
                    continue
                res[ax] += c * np.roll(scalar_field, rolls[n], axis=ax)
            res[ax] /= h ** order
        return res

    @callable_to_array
    @has_default_accuracy
    @takes_scalar_field
    def grad(self, scalar_field: Union[Callable, np.ndarray], accuracy: int = None) -> np.array:
        """
        Gradient of a scalar field.

        Args:
            scalar_field (function/numpy.ndarray): A function taking this `RectMesh` object and returning a scalar
                field, or the scalar field as an array.
            accuracy (int): The order of approximation in grid spacing. See `central_difference_table` for all choices.
                (Default is None, which falls back on the class attribute of the same name.)

        Returns:
            (numpy.ndarray): The vector field gradient of the scalar input at each point on the mesh in each dimension.
                E.g. for a scalar field with shape `(nx, ny, nz)`, returns shape `(3, nx, ny, nz)`.
        """
        return self.derivative(scalar_field, order=1, accuracy=accuracy)

    @callable_to_array
    @has_default_accuracy
    @takes_vector_field
    def div(self, vector_field: Union[Callable, np.ndarray], accuracy: int = None) -> np.array:
        """
        Divergence of a vector field.

        Args:
            vector_field (function/numpy.ndarray): A function taking this `RectMesh` object and returning a vector
                field, or the vector field as an array.
            accuracy (int): The order of approximation in grid spacing. See `central_difference_table` for all choices.
                (Default is None, which falls back on the class attribute of the same name.)

        Returns:
            (numpy.ndarray): The scalar field divergence of the vector input at each point on the mesh.
        """
        return np.sum([self.derivative(vector_field[ax], accuracy=accuracy)[ax] for ax in np.arange(self.dim)], axis=0)

    @callable_to_array
    @has_default_accuracy
    @takes_scalar_field
    def laplacian(self, scalar_field: Union[Callable, np.ndarray], accuracy: int = None) -> np.array:
        """
        Discrete Laplacian operator applied to a given function or scalar field.

        Args:
            scalar_field (function/numpy.ndarray): A function taking this `RectMesh` object and returning a scalar
                field, or the scalar field as an array.
            accuracy (int): The order of approximation in grid spacing. See `central_difference_table` for all choices.
                (Default is None, which falls back on the class attribute of the same name.)

        Returns:
            (numpy.ndarray): The scalar field Laplacian of the scalar input at each point on the mesh.
        """
        return self.derivative(scalar_field, order=2, accuracy=accuracy).sum(axis=0)

    @callable_to_array
    @has_default_accuracy
    @takes_vector_field
    def curl(self, vector_field: Union[Callable, np.ndarray], accuracy: int = None) -> np.array:
        """
        Curl of a vector field.

        Note: Only works for 3d vector fields!

        Args:
            vector_field (function/numpy.ndarray): A function taking this `RectMesh` object and returning a 3d vector
                field, or the 3d vector field as an array.
            accuracy (int): The order of approximation in grid spacing. See `central_difference_table` for all choices.
                (Default is None, which falls back on the class attribute of the same name.)

        Returns:
            (numpy.ndarray): The vector field curl of the vector input at each point on the mesh in all three
                dimensions. I.e. for a vector field with shape `(3, nx, ny, nz)`, returns shape `(3, nx, ny, nz)`.

        Raises:
            (NotImplementedError): If the vector field provided is not three dimensional.
        """
        if self.dim != 3:
            raise NotImplementedError("I'm no mathematician, so curl is only coded for the traditional 3d space.")
        grads = np.array([self.derivative(vf, accuracy=accuracy) for vf in vector_field])
        pos = np.array([grads[(2 + i) % self.dim][(1 + i) % self.dim] for i in range(self.dim)])
        neg = np.array([grads[(1 + i) % self.dim][(2 + i) % self.dim] for i in range(self.dim)])
        return pos - neg
