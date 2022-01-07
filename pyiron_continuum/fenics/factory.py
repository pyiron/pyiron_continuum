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

with ImportAlarm("precice-fenics workflows require:"
                 "- fenicsprecice") as precice_alarm:
    import fenicsprecice

from pyiron_base import PyironFactory

__author__ = "Liam Huber"
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
    def __init__(self):
        super().__init__()
        self._regular = RegularMeshFactory()
        self._unit = UnitMeshFactory()

    @property
    def regular_mesh(self):
        return self._regular

    @property
    def unit_mesh(self):
        return self._unit

    def circle(self, center, radius):
        return mshr.Circle(FEN.Point(*center), radius)
#    circle.__doc__ = mshr.Circle.__doc__

    def square(self, length, origin=None):
        if origin is None:
            x, y = 0, 0
        else:
            x, y = origin[0], origin[1]
        return mshr.Rectangle(FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y))
#    square.__doc__ = mshr.Rectangle.__doc__

    @staticmethod
    def box(corner1=None, corner2=None):
        """A 3d rectangular prism from `corner1` to `corner2` ((0, 0, 0) to (1, 1, 1) by default)"""
        corner1 = corner1 or (0, 0, 0)
        corner2 = corner2 or (1, 1, 1)
        return mshr.Box(FEN.Point(corner1), FEN.Point(corner2))

    @staticmethod
    def tetrahedron(p1, p2, p3, p4):
        """A tetrahedron defined by four points. (Details to be discovered and documented.)"""
        return mshr.Tetrahedron(FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4))


class UnitMeshFactory(PyironFactory):
    def square(self, nx, ny):
        return FEN.UnitSquareMesh(nx, ny)
 #   square.__doc__ = FEN.UnitSquareMesh.__doc__


class RegularMeshFactory(PyironFactory):
    @staticmethod
    def rectangle(p1, p2, nx, ny, **kwargs):
        return FEN.RectangleMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, **kwargs)
#    rectangle.__doc__ = FEN.RectangleMesh.__doc__

    @staticmethod
    def box(p1, p2, nx, ny, nz):
        return FEN.BoxMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, nz)
 #   box.__doc__ = FEN.BoxMesh.__doc__


class BoundaryConditionFactory(PyironFactory):
    def __init__(self, job):
        self._job = job

    @staticmethod
    def _default_bc_fnc(x, on_boundary):
        return on_boundary

    def dirichlet(self, expression, bc_fnc=None):
        """
        This function defines Dirichlet boundary condition based on the given expression on the boundary.

        Args:
            expression (string): The expression used to evaluate the value of the unknown on the boundary.
            bc_fnc (fnc): The function which evaluates which nodes belong to the boundary to which the provided
                expression is applied as displacement.
        """
        bc_fnc = bc_fnc or self._default_bc_fnc
        return FEN.DirichletBC(self._job.V, expression, bc_fnc)


class FenicsSubDomain(FEN.SubDomain):
    def __init__(self, conditions, tol=1E-14):
        super(FenicsSubDomain, self).__init__()
        self._conditions = conditions

    def _evalConditions(self, x):
        if eval(self._conditions):
            return True
        else:
            return False

    def inside(self, x, onboundary):
        if onboundary:
            return self._evalConditions(x)

class SubDomainFactory(PyironFactory):
    @staticmethod
    def inside(conditions):
        return FenicsSubDomain(conditions)

class PreciceAdapter(fenicsprecice.Adapter):
     def __init__(self, job, config_file):
         self._job = job
         super(PreciceAdapter, self).__init__(adapter_config_filename=config_file)
         self._coupling_bc = None
         self._write_object = None
         self._function_space= self._job.V


     def initialize(self, coupling_boundary, write_object, function_space=None):
         if function_space is None:
             function_space= self._function_space
         return super().initialize(coupling_boundary, function_space, write_object)

