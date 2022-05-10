# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Factories for Fenics-related object creation.
"""

from pyiron_base import ImportAlarm, HasStorage
import re
with ImportAlarm(
    "fenics functionality requires the `fenics`, `mshr` modules (and their dependencies) specified as extra"
    "requirements. Please install it and try again."
) as fenics_alarm:
    import fenics as FEN
    import mshr
    from fenics import (
        near, Constant, Expression, Point, BoxMesh, RectangleMesh, UnitSquareMesh, UnitCubeMesh, UnitDiscMesh,
        UnitTriangleMesh, UnitIntervalMesh
    )
    from mshr import generate_mesh, Tetrahedron, Box, Rectangle, Circle
    import dolfin.cpp.mesh as FenicsMesh
from pyiron_base import PyironFactory
from abc import ABC, abstractmethod

__author__ = "Liam Huber, Muhammad Hassani, Niklas Siemer"
__copyright__ = (
    "Copyright 2020, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Dec 26, 2020"


class StringInputParser(ABC):
    def __init__(self, input_string: str, __verbose=True, **kwargs):
        self._splitting_elements = [r"\*?\*", r"\+", r"-", r"/", r"\(", r"\)", r","]
        self._verbose = __verbose
        self._test_kwargs(kwargs)
        self._test_elements(input_string, kwargs)

    @property
    @abstractmethod
    def _known_elements(self):
        pass

    def _test_kwargs(self, kwargs):
        failures = {}
        for key, value in kwargs.items():
            try:
                float(value)
            except:
                try:
                    self.__class__(value, __verbose=False)
                except:
                    failures[key] = value
        if failures and self._verbose:
            raise ValueError(f"Got an unexpected kwarg(s) '{failures}'")

    def _split_input(self, input_string):
        scientific_notation_re = r"[0-9]*\.[0-9]+[eE][+-]?[0-9]+|[0-9]+[eE][+-]?[0-9]+"
        return [
            r.strip()
            for r in re.split(
                "|".join(self._splitting_elements),
                "".join(re.split(scientific_notation_re, input_string)),
            )
            if len(r.strip()) > 0
        ]

    def _test_elements(self, input_string, kwargs):
        failures = []
        for e in self._split_input(input_string):
            if e in self._known_elements:
                continue
            elif e.isnumeric():
                continue
            elif e in kwargs.keys():
                continue
            elif e[0] == 'x' and e[1] == '[' and e[-1] == ']':
                # TODO: This is a crap test hard wired for the boundary conditions, but we need to fix the regex to
                #  allow known elements to be indexed...
                continue
            else:
                try:
                    float(e)
                    continue
                except:
                    failures.append(e)
        if failures and self._verbose:
            raise ValueError(f"Got an unexpected symbol(s) '{failures}'")


class BCParser(StringInputParser):
    @property
    def _known_elements(self):
        return ["x", "near", "Constant", "Expression", "and", "or", ">", "<"]


class MeshParser(StringInputParser):
    @property
    def _known_elements(self):
        return [
            "Point", "BoxMesh", "RectangleMesh", "UnitSquareMesh", "UnitCubeMesh", "UnitDiscMesh", "UnitTriangleMesh",
            "UnitIntervalMesh",
            "generate_mesh", "Tetrahedron", "Box", "Rectangle", "Circle"
        ]


class SerialBoundaries(PyironFactory, HasStorage):
    def __init__(self):
        PyironFactory.__init__(self)
        HasStorage.__init__(self)
        self.storage.pairs = []

    def add(self, value, condition):
        # BCParser(value)
        # BCParser(condition)
        self.storage.pairs.append((value, condition))

    def list(self):
        return self.storage.pairs

    def clear(self):
        self.storage.pairs = []

    def __call__(self, function_space):
        fenics_objects = []
        for (v, w) in self.list():
            def boundary_func(x, on_boundary):
                try:
                    return on_boundary and eval(w)
                except Exception as err_msg:
                    print(err_msg)
            fenics_objects.append(FEN.DirichletBC(function_space, eval(v), boundary_func))
        return fenics_objects


class SerialMesh(PyironFactory, HasStorage):
    def __init__(self):
        PyironFactory.__init__(self)
        HasStorage.__init__(self)
        self._mesh = None
        self.storage.expression = 'BoxMesh(p1, p2, nx, ny, nz)'
        self.storage.kwargs = {'p1': 'Point((0,0,0))', 'p2': 'Point((1, 1, 1))', 'nx': 2, 'ny': 2, 'nz': 2}

    def from_string(self, expression, **kwargs):
        MeshParser(expression, **kwargs)
        self.storage.expression = expression
        self.storage.kwargs = kwargs
        self._mesh = self.generate()

    def generate(self):
        for k, v in self.storage.kwargs.items():
            exec(f'{k} = {v}')
        mesh = eval(self.storage.expression)
        if not isinstance(mesh, FenicsMesh.Mesh):
            raise TypeError(f'Expected the mesh to be of type dolfin.cpp.mesh.Mesh, but got {type(mesh)}. You may '
                            f'need to wrap your expression in "generate_mesh()" to convert a mshr mesh to a dolphin '
                            f'mesh.')
        return mesh

    def __call__(self):
        if self._mesh is None:
            self._mesh = self.generate()
        return self._mesh


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

    def __init__(self, job):
        super().__init__()
        self._job = job
        self._subdomain_dict = {}
        self._bcs = []
        self._bc = BoundaryConditionFactory(job=self._job)
        self._mesh = GeneralMeshFactory(job=self._job)

    def clear_boundaries(self):
        """Clear all existing boundary conditions in the conditions list."""
        self._bcs = []

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

    def boundary(
        self,
        bc_type,
        expression=None,
        constant=None,
        bc_func=None,
        subdomain_name=None,
        **kwargs,
    ):
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
            NotSetCorrectlyError(
                "the domain factory is not set correctly! "
                "please use job.domain.append_boundary(..)"
            )

        if bc_type is None:
            NotSetCorrectlyError(
                "the domain factory is not set correctly! "
                "please use job.domain.append_boundary(..)"
            )
        if subdomain_name:
            subdomain = self._subdomain_dict[subdomain_name]
        else:
            subdomain = None
        if bc_type.lower() == "dirichlet":
            if constant and expression:
                raise ValueError(
                    "The dirichlet boundary whether is set by "
                    "a constant value or an expression"
                )
            elif isinstance(expression, str):
                expression = FEN.Expression(expression, **kwargs)
            elif constant is not None:
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


class GeneralMeshFactory(PyironFactory):
    def __init__(self, job):
        super().__init__()
        self._regular = RegularMeshFactory(job)
        self._unit = UnitMeshFactory(job)
        self._job = job
        self._mshr_domain = None

    def __call__(self):
        return self._job._mesh

    @property
    def regular(self):
        return self._regular

    @property
    def unit(self):
        return self._unit

    def circle(self, center, radius, inplace=True):
        if inplace:
            self._job._set_mesh(mshr.Circle(FEN.Point(*center), radius))
        else:
            return mshr.Circle(FEN.Point(*center), radius)

    def square(self, length, origin=None, inplace=True):
        if origin is None:
            x, y = 0, 0
        else:
            x, y = origin[0], origin[1]
        if inplace:
            self._job._set_mesh(
                mshr.Rectangle(
                    FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y)
                )
            )
        else:
            return mshr.Rectangle(
                FEN.Point(0 + x, 0 + y), FEN.Point(length + x, length + y)
            )

    def box(self, corner1=None, corner2=None, inplace=True):
        """A 3d rectangular prism from `corner1` to `corner2` ((0, 0, 0) to (1, 1, 1) by default)"""
        corner1 = corner1 or (0, 0, 0)
        corner2 = corner2 or (1, 1, 1)
        if inplace:
            self._job._set_mesh(mshr.Box(FEN.Point(corner1), FEN.Point(corner2)))
        else:
            return mshr.Box(FEN.Point(corner1), FEN.Point(corner2))

    def tetrahedron(self, p1, p2, p3, p4, inplace=True):
        """A tetrahedron defined by four points. (Details to be discovered and documented.)"""
        if inplace:
            self._job._set_mesh(
                mshr.Tetrahedron(
                    FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4)
                )
            )
        else:
            return mshr.Tetrahedron(
                FEN.Point(p1), FEN.Point(p2), FEN.Point(p3), FEN.Point(p4)
            )

    def generate(self, domain):
        self._job._set_mesh(domain)


class UnitMeshFactory(PyironFactory):
    def __init__(self, job):
        super(UnitMeshFactory, self).__init__()
        self._job = job

    def square(self, nx, ny):
        self._job._mesh = FEN.UnitSquareMesh(nx, ny)


#   square.__doc__ = FEN.UnitSquareMesh.__doc__


class RegularMeshFactory(PyironFactory):
    def __init__(self, job):
        super(RegularMeshFactory, self).__init__()
        self._job = job

    def rectangle(self, p1, p2, nx, ny, **kwargs):
        self._job._mesh = FEN.RectangleMesh(
            FEN.Point(p1), FEN.Point(p2), nx, ny, **kwargs
        )

    #    rectangle.__doc__ = FEN.RectangleMesh.__doc__

    def box(self, p1, p2, nx, ny, nz):
        self._job._mesh = FEN.BoxMesh(FEN.Point(p1), FEN.Point(p2), nx, ny, nz)


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
            raise ValueError(
                "can not have both bc_func and subdomain set at the same time"
            )
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


class SolverConfig:
    """
    The class defines the equation to be solved by fenics.solve() function
    The job.solver defines:
        - V (FunctionSpace)
        - u (TrialFunction)
        - v(TestFunction)
        - V_g(VectorFunctionSpace)
        - F(the equation in the form of F=0)
        - u_n(the initial contdition, it is normally a projected expression
           or an interpolated expression over the function space),
        - f(a source/sink term which can be a constant or an expression)
        - rhs (right hand side of the equation, normally found by calling FEN.rhs(F)
        - lhs (left hand side of the equation, normally found by calling FEN.lhs(F)
        - dt (FEN.Constant, or a float number) It is set initially to job.input.dt
        - dx (fenics.dx)
        - dot (fenics.dot)
        - inner (fenics.inner)
        - grad (fenics.grad)
        - time_dependent_expressions (a list of time dependent expressions)
    Args:
        job(pyiron_continuum.fenics): a fenics job

    An example of the usage could be:
    >>> u_analytical = job_d.Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
    >>> job_d.solver.set_initial_condition(interpolate=True, expression=u_analytical)
    >>> job_d.solver.f = job_d.Constant(u_analytical.beta - 2 - 2*u_analytical.alpha)
    >>> job_d.solver.F = 'u * v / dt * dx + dot(grad(u), grad(v)) * dx - (u_n / dt + f) * v * dx'
    >>> job_d.solver.time_dependent_expressions.append(u_analytical)

    """

    def __init__(self, job, func_space_class=FEN.FunctionSpace):
        self._job = job
        if self._job._mesh is None:
            raise NotSetCorrectlyError(
                "Before accessing job.solver, job._mesh should be defined"
                "You can use job.domain.mesh to create job._mesh"
            )
        else:
            self._V = func_space_class(
                job._mesh, job.input.element_type, job.input.element_order
            )  # finite element volume space
            if job.input.element_order > 1:
                self._V_g = FEN.VectorFunctionSpace(
                    job._mesh, job.input.element_type, job.input.element_order - 1
                )
            else:
                self._V_g = FEN.VectorFunctionSpace(
                    job._mesh, job.input.element_type, job.input.element_order
                )

            self._u = FEN.TrialFunction(self._V)  # u is the unkown function
            self._v = FEN.TestFunction(self._V)  # the test function
            self._F = None
            self._lhs = None
            self._rhs = None
            self._u_n = None
            self._string_equation = None
            self._f = None
            if self._job.input.dt:
                self.dt = self._job.input.dt
            else:
                self.dt = 1.0
            self._extra_parameters = {}
            self._extra_func = {}
            self._solution = FEN.Function(self._V)
            self._flux = FEN.Function(self._V_g)
            self.time_dependent_expressions = []
            self.assigned_u = None
            self._update_equation_func = None
            self._update_equation_func_args = None
            self._accepted_keys = ["f", "u_n"]

    def set_extra_func(self, func_key, func):
        self._extra_func[func_key] = func

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, _val):
        self._dt = _val

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, expression):
        if isinstance(expression, FEN.Expression) or isinstance(
            expression, FEN.Constant
        ):
            self._f = expression
        else:
            raise ValueError(
                "job.solver.f must be of type fenics.Expression or fenics.Constant"
            )

    def set_expression(self, key, constant=None, expression=None, **kwargs):
        if key in self._accepted_keys:
            if expression:
                exec(f"self.{key} = FEN.Expression({expression}, **{kwargs})")
            elif constant:
                if isinstance(constant, str):
                    for k, val in kwargs.items():
                        exec(f"{k} = val")
                        exec("u_m = val")
                    constant_value = eval(constant)
                    exec(f"self.{key} = FEN.Constant({constant_value})")
                else:
                    exec(f"self.{key} = FEN.Constant({constant})")
            else:
                raise ValueError(
                    f"for initializing {key}, one should use either a constant or expression"
                )
        else:
            self._extra_parameters[key] = [expression, kwargs]

    def set_update_func(self, func):
        self._update_equation_func = func

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, equation):
        if isinstance(equation, str):
            self._string_equation = equation
            self._evaluate_equation()
        else:
            self._F = equation
            self.lhs = FEN.lhs(self._F)
            self.rhs = FEN.rhs(self._F)

    def _evaluate_equation(self):
        for key, val in self._extra_parameters.items():
            kwargs = ""
            counter = 0
            for k, v in val[1].items():
                if counter < len(val[1]) - 1:
                    kwargs += f"{k}={v},"
                else:
                    kwargs += f"{k}={v}"
                counter += 1
            exec(f'{key} = FEN.Expression("{val[0]}", {kwargs})')

        for key, val in self._extra_func.items():
            exec(f"{key}=val")

        u = self.u
        v = self._v
        u_n = self._u_n
        dot = FEN.dot
        grad = FEN.grad
        dx = FEN.dx
        dt = self._dt
        inner = FEN.inner
        # update_func = self._update_equation_func
        for key in self._accepted_keys:
            if not eval(f"self.{key}") is None:
                exec(f"{key}= self.{key}")

        try:
            self._F = eval(self._string_equation)
            self.lhs = FEN.lhs(self._F)
            self.rhs = FEN.rhs(self._F)
        except Exception as err_msg:
            raise Exception(err_msg)

    @property
    def u_n(self):
        return self._u_n

    @u_n.setter
    def u_n(self, expression, **kwargs):
        if not isinstance(expression, FEN.Expression):
            try:
                expression = FEN.Expression(expression, **kwargs)
            except Exception as err_msg:
                raise ValueError(err_msg)
        self._u_n = expression

    def set_initial_condition(
        self,
        function=None,
        expression=None,
        interpolate=True,
        project=False,
        function_space=None,
        **kwargs,
    ):
        """
        This function set job.solver.u_n as the initial value
        It is normally set to an interpolated or projected form
        of an expression over the given function space
        Args:
            - expression(str or fenics.Expression): the expression used for interpolation
            - function(fenic.Function): an already interplated or
            projected function over the given function space
            - function_space: in the case that the interpolation
            or projection should be done via a non-default FunctionSpcae
            - kwargs: the key-value arguements, in case that the given expression is a string.
                     In this case, FEN.Expression(expression, **kwargs) is called.
        """
        print("The intial value will be assigned to 'u_n'!")
        if function is None and expression is None:
            ValueError("At least function or expression must be specified")

        if interpolate:
            if isinstance(expression, str):
                expression = FEN.Expression(expression, **kwargs)
            self._u_n = self.interpolate(expression, function_space)
        elif project:
            NotImplementedError("The project mode, is not yet implemented")

    def interpolate(self, expression, function_space=None, **kwargs):
        """
        Interpolate an expression over the given function space.

        Args:
            - expression (str or FEN.Expression): The function to interpolate.
            - function_space: if the default functionSpace job.V is not
            meant to be used, you can provide your own
            - kwargs: the key-value arguements, in case that the given expression is a string.
                    In this case, FEN.Expression(expression, **kwargs) is called.

        Returns:
            (?): Interpolated function.
        """

        if function_space is None:
            function_space = self._V

        try:
            if isinstance(expression, str):
                expression = FEN.Expression(expression, **kwargs)
        except Exception as err_msg:
            raise ValueError(err_msg)

        return FEN.interpolate(expression, function_space)

    def update_lhs_rhs(self):
        self.lhs = FEN.lhs(self.F)
        self.rhs = FEN.rhs(self.F)

    @property
    def lhs(self):
        return self._lhs

    @lhs.setter
    def lhs(self, _lhs):
        if isinstance(_lhs, str):
            self._lhs = self._evaluate_strings(_lhs)
        else:
            self._lhs = _lhs

    @property
    def rhs(self):
        return self._rhs

    @rhs.setter
    def rhs(self, _rhs):
        if isinstance(_rhs, str):
            self._rhs = self._evaluate_strings(_rhs)
        else:
            self._rhs = _rhs

    @property
    def V_g(self):
        return self._V_g

    @property
    def flux(self):
        return self._flux

    @flux.setter
    def flux(self, val):
        self._flux = val

    def calc_flux(self):
        """
        It calculates the flux, based on the given solution field
        """
        w = FEN.TrialFunction(self._V_g)
        v = FEN.TestFunction(self._V_g)
        a = FEN.inner(w, v) * FEN.dx
        L = FEN.inner(FEN.grad(self.solution), v) * FEN.dx
        FEN.solve(a == L, self.flux)

    @property
    def V(self):
        return self._V

    @property
    def v(self):
        return self._v

    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, _func):
        self._u = _func

    @property
    def u_n(self):
        return self._u_n

    @property
    def solution(self):
        return self._solution

    @property
    def V_g(self):
        return self._V_g

    def _evaluate_strings(self, to_be_evaluated):
        if self.assigned_u is None and not self._u_n is None:
            self.assigned_u = self._u_n
        elif self._job.input.n_steps > 1 and self.assigned_u is None:
            raise ValueError(
                "When the timesteps are larger that one, job.solver.assigned_u or self.solver.u_n should be specified"
            )

        for key, val in self._extra_parameters.items():
            kwargs = ""
            counter = 0
            for k, v in val[1].items():
                if counter < len(val[1]) - 1:
                    kwargs += f"{k}={v},"
                else:
                    kwargs += f"{k}={v}"
                counter += 1
            exec(f'{key} = FEN.Expression("{val[0]}", {kwargs})')

        for key, val in self._extra_func.items():
            exec(f"{key}=val")

        u = self._u
        v = self._v
        u_n = self._u_n
        dot = FEN.dot
        grad = FEN.grad
        dx = FEN.dx
        dt = self._dt
        inner = FEN.inner
        Constant = FEN.Constant
        for key in self._accepted_keys:
            if not eval(f"self.{key}") is None:
                exec(f"{key}= self.{key}")
        try:
            return eval(to_be_evaluated)

        except Exception as err_msg:
            raise Exception(err_msg)

    @staticmethod
    def expression(expression, **kwargs):
        try:
            return FEN.Expression(expression, **kwargs)
        except Exception as err_msg:
            raise Exception(err_msg)

    def error_norm(self, expression, **kwargs):
        if isinstance(expression, str):
            expression = FEN.Expression(expression, **kwargs)
        return FEN.errornorm(expression, self.solution, "L2")


class NotSetCorrectlyError(Exception):
    "raised when the object is not configured correctly!"
    pass
