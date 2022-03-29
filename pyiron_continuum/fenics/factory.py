# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Factories for Fenics-related object creation.
"""

from pyiron_base import ImportAlarm
with ImportAlarm(
        'fenics functionality requires the `fenics`, `mshr` modules (and their dependencies) specified as extra'
        'requirements. Please install it and try again.'
) as fenics_alarm:
    import fenics as FEN
    import mshr
    from fenics import near
from pyiron_base import PyironFactory

__author__ = "Liam Huber, Muhammad Hassani"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Dec 26, 2020"


class DomainFactory(PyironFactory):
    """
    The domain factory provides access to mesh generation, addition of boundary conditions,
    and/or subdomains
    Creation of mesh:
    >>> job.domain.mesh.unit.rectangle(...)
    or
    >>> job.domain.mesh.regular.rectangle(...)
    Creation of subdomain:
    >>> job.domain.subdomain(name, condition)
    The subdomains are stored in a dictionary, where the names are used as keys
    then to obtain a subdomain with a given name, you can:
    >>> job.domain.get_subdomain(name)
    To add a boundary condition:
    >>> job.domain.boundary(bc_type, ...)
    Currently, only bc_type="Dirichlet" is supported
    To get the list of the boundary conditions, you can use:
    >>> job.domain.boundaries_list
    To append to the list of boundaries:
    >>> job.domain.append_boundary(given_boundary_condition)

    """
    def __init__(self, job=None):
        super().__init__()
        self._regular = RegularMeshFactory(job)
        self._unit = UnitMeshFactory(job)
        self._job = job
        self._subdomain_dict = {}
        self._bcs = []
        self._bc = BoundaryConditionFactory(job=self._job)
        self._mesh = GeneralMeshFactory(job=self._job)

    def get_subdomain(self, name):
        """
        returns a subdomain with a given name
        name(str): name of the desired subdomain
        """
        try:
            return self._subdomain_dict[name]
        except Exception as err_msg:
            raise Exception(err_msg)

    def list_subdomains(self):
        """
        returns the dictionary of the subdomains
        """
        return self._subdomain_dict

    def subdomain(self, name, conditions):
        """
        This adds a subdomain with the given name to the dictionary of the subdomains
        Args:
             - name(str): a given name
             - conditions(str): a condition describing the geometry of the boundary
        An example:
        >>> some_condition = 'not near(x[0], 1.0, 1E-14) \
                                    or near(x[1], 1.0, 1E-14) or near(x[1], 0., 1E-14)'
        >>> job.domain.subdomain(name='some name', conditions=some_condition)
        """
        self._subdomain_dict[name] = FenicsSubDomain(conditions=conditions)

    def boundary(self, bc_type, expression=None, constant = None, bc_func=None, subdomain_name=None, **kwargs):
        """
        This adds a boundary of the type bc_type based on the given expression or bc_function
        passing a subdomain_name is optional
        Args:
            - bc_type(str): type of boundary condition, currently only Dirichlet is supported
            - expression(str or FEN.Expression): an expression describing the boundary
            - subdomain_name(str): the name of subdomain which is stored in job.domain.list_subdomain
            - bc_func(func): a python function which describes the boundary, example would be:
                def boundary(x, on_boundary):
                    return on_boundary
            - kwargs: the key-value arguements, in case that the given expression is a string.
                     In this case, FEN.Expression(expression, **kwargs) is called.
        """
        if self._job is None:
            NotSetCorrectlyError('the domain factory is not set correctly! '
                                 'please use job.domain.append_boundary(..)')

        if bc_type is None:
            NotSetCorrectlyError('the domain factory is not set correctly! '
                                 'please use job.domain.append_boundary(..)')
        if subdomain_name:
            subdomain = self._subdomain_dict[subdomain_name]
        else:
            subdomain = None
        if bc_type in ['dirichlet', 'Dirichlet']:
            if constant and expression:
                raise ValueError("The dirichlet boundary whether is set by "
                                 "a constant value or an expression")
            elif isinstance(expression, str):
                expression = FEN.Expression(expression, **kwargs)
            elif not constant is None:
                expression = FEN.Constant(constant)

            self._bcs.append(self._bc.dirichlet(expression, bc_func, subdomain))

    def append_bc(self, new_bc):
        """
        this appends the new_bc to the list of boundary conditions
        Args:
            new_bc: a fenics boundary condition
        """
        try:
            self._bcs.append(new_bc)
        except Exception as err_msg:
            raise Exception(err_msg)

    @property
    def boundaries_list(self):
        """
        list of boundary conditions
        """
        return self._bcs

    @property
    def mesh(self):
        return self._mesh

    @property
    def regular_mesh(self):
        return self._regular

    @property
    def unit_mesh(self):
        return self._unit

    def circle(self, center, radius):
        if not self._job:
            return mshr.Circle(FEN.Point(*center), radius)
        else:
            self._job.mesh = mshr.Circle(FEN.Point(*center), radius)
#    circle.__doc__ = mshr.Circle.__doc__

    def square(self, length, origin=None):
        if origin is None:
            x, y = 0, 0
        else:
            x, y = origin[0], origin[1]
        if not self._job:
            return mshr.Rectangle(FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y))
        else:
            self._job.mesh = mshr.Rectangle(FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y))
#    square.__doc__ = mshr.Rectangle.__doc__


    def box(self, corner1=None, corner2=None):
        """A 3d rectangular prism from `corner1` to `corner2` ((0, 0, 0) to (1, 1, 1) by default)"""
        corner1 = corner1 or (0, 0, 0)
        corner2 = corner2 or (1, 1, 1)
        if not self._job:
            return mshr.Box(FEN.Point(corner1), FEN.Point(corner2))
        else:
            self._job.mesh = mshr.Box(FEN.Point(corner1), FEN.Point(corner2))

    def tetrahedron(self, p1, p2, p3, p4):
        """A tetrahedron defined by four points. (Details to be discovered and documented.)"""
        if not self._job:
            return mshr.Tetrahedron(FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4))
        else:
            self._job.mesh = mshr.Tetrahedron(FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4))


class GeneralMeshFactory(PyironFactory):
    def __init__(self, job=None):
        super().__init__()
        self._regular = RegularMeshFactory(job)
        self._unit = UnitMeshFactory(job)
        self._job = job

    def __call__(self):
        return self._job._mesh

    @property
    def regular(self):
        return self._regular

    @property
    def unit(self):
        return self._unit

    def circle(self, center, radius):
        if not self._job:
            return mshr.Circle(FEN.Point(*center), radius)
        else:
            self._job.mesh = mshr.Circle(FEN.Point(*center), radius)
    #    circle.__doc__ = mshr.Circle.__doc__

    def square(self, length, origin=None):
        if origin is None:
            x, y = 0, 0
        else:
            x, y = origin[0], origin[1]
        if not self._job:
            return mshr.Rectangle(FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y))
        else:
            self._job.mesh = mshr.Rectangle(FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y))
    #    square.__doc__ = mshr.Rectangle.__doc__

    def box(self, corner1=None, corner2=None):
        """A 3d rectangular prism from `corner1` to `corner2` ((0, 0, 0) to (1, 1, 1) by default)"""
        corner1 = corner1 or (0, 0, 0)
        corner2 = corner2 or (1, 1, 1)
        if not self._job:
            return mshr.Box(FEN.Point(corner1), FEN.Point(corner2))
        else:
            self._job.mesh = mshr.Box(FEN.Point(corner1), FEN.Point(corner2))

    def tetrahedron(self, p1, p2, p3, p4):
        """A tetrahedron defined by four points. (Details to be discovered and documented.)"""
        if not self._job:
            return mshr.Tetrahedron(FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4))
        else:
            self._job.mesh = mshr.Tetrahedron(FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4))

class UnitMeshFactory(PyironFactory):
    def __init__(self, job=None):
        super(UnitMeshFactory, self).__init__()
        self._job = job

    def square(self, nx, ny):
        if not self._job:
            return FEN.UnitSquareMesh(nx, ny)
        else:
            self._job.mesh = FEN.UnitSquareMesh(nx, ny)
 #   square.__doc__ = FEN.UnitSquareMesh.__doc__


class RegularMeshFactory(PyironFactory):
    def __init__(self, job=None):
        super(RegularMeshFactory, self).__init__()
        if job is not None:
            self._job = job

    def rectangle(self, p1, p2, nx, ny, **kwargs):
        if not self._job:
            return FEN.RectangleMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, **kwargs)
        else:
            self._job.mesh = FEN.RectangleMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, **kwargs)
#    rectangle.__doc__ = FEN.RectangleMesh.__doc__

    def box(self, p1, p2, nx, ny, nz):
        if not self._job:
            return FEN.BoxMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, nz)
        else:
            self._job.mesh = FEN.BoxMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, nz)
 #   box.__doc__ = FEN.BoxMesh.__doc__


class BoundaryConditionFactory(PyironFactory):
    def __init__(self, job):
        self._job = job

    @staticmethod
    def _default_bc_fnc(x, on_boundary):
        return on_boundary

    def dirichlet(self, expression, bc_fnc=None, subdomain=None):
        """
        This function defines Dirichlet boundary condition based on the given expression on the boundary.

        Args:
            expression (string): The expression used to evaluate the value of the unknown on the boundary.
            bc_fnc (fnc): The function which evaluates which nodes belong to the boundary to which the provided
                expression is applied as displacement.
        """
        if not bc_fnc is None and not subdomain is None:
            raise ValueError('can not have both bc_func and subdomain set at the same time')
        elif not bc_fnc is None:
            bc_fnc = bc_fnc
            return FEN.DirichletBC(self._job.solver.V, expression, bc_fnc)
        elif not subdomain is None:
            return FEN.DirichletBC(self._job.solver.V, expression, subdomain)
        else:
            return FEN.DirichletBC(self._job.solver.V, expression, self._default_bc_fnc)


class FenicsSubDomain(FEN.SubDomain):
    """
    The refactory of fenics.SubDomain, which creates a subdomain
    based on the provided conditions
    Args:
        conditions(str): a condition describing the subdomain
    Example
        >>> job.domain.subdomain(condition='near(x[0], 1.0, 1E-14)')
    """
    def __init__(self, conditions):
        super(FenicsSubDomain, self).__init__()
        self._conditions = conditions

    def _evalConditions(self, x):
        if eval(self._conditions):
            return True
        else:
            return False

    def inside(self, x, onboundary):
        if onboundary and self._conditions:
            return self._evalConditions(x)
        else:
            False


class NotSetCorrectlyError(Exception):
    "raised when the object is not configured correctly!"
    pass
