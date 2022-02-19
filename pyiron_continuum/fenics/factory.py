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
    import dolfin

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
    """
    The domain factory provides access to mesh generation, addition of boundary conditions,
    and addition of subdomains
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

    def boundary(self, bc_type, expression=None, bc_func=None, subdomain_name=None, **kwargs):
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
            if isinstance(expression, str):
                expression = FEN.Expression(expression, **kwargs)
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


class PreciceConf(PyironFactory):
    """
    What you need to do in most cases:
        job.init_adapter_conf(config_file='path_to_json_file', coupling_boundary_subdomain='name of the coupling subdomain',
                              write_object=some_FEN.Function)
        job.adapter_conf.predefined_coupling(coupling_type='Dirichlet', flux_direction='x')
    or in case of Neumann boundary:
       job_n.adapter_conf.predefined_coupling(coupling_type="Neumann")

    job.adapter_conf allows you to define functions which updates couplings during the job.run().
    There are two presets: 'Dirichlet and Neumann"
    if you want to use these two presets: you just need to:
    job.adapter_config.predefined_coupling(coupling_type, flux_direction=None)
    coupling type can be "Dirichlet" or "Neumann"
    if you use "Dirichlet", you need to pass
    in the direction of flux, 'x', 'y', or 'z'

    If you need another customized coupling boundary condition,
    you can pass some functions to update the coupling during job.run()
    The two functions needed are:

        job.adapter_conf.update_boundary_func = some_user_defined_funct ;
        job.adapter.coupling_data_func=some_user_defined_funct

    The first function returns a boundary condition,
    which adds to the list of the boundary conditions.
    It is important that the function needs three arguments:
        - job
        - coupling_expression
        - coupling_subdomain
    An example of such a function, which adds a dirichlet boundary:
        def some_user_defined_func(job, coupling_expression,
                                    coupling_subdomain)
            job.domain.boundary(bc_type='Dirichlet',
                                coupling_expression, coupling_subdomain)

    The second function returns the value which is
    passed to the adapter.write_data()
    The function requires two arguments:
        - config: this represents job.adapter_conf
        - job: this represents the job itself
    An example of such a function:
        def some_user_defined_func(config, job):
            return job.solution

    Args:
        config_file(str): the path to the adapter configuration json file
        coupling_boundary_subdomain(str): the name of the coupling subdomain;
            you can get the list of subdomain by job.domain.list_subdomains
        write_object(FEN.Function): the write_object required
            by the precice fenics adapter
        function_space(FEN.FunctionSpace): the read_function_space required
            by the precice fenics adapter; if not given the configuration assumes
            read_function_space = job.solver.V
    To understand better how the fenics-precice adapter works,
    take a look at the example here:
    https://github.com/precice/tutorials/blob/master/partitioned-heat-conduction/fenics/heat.py
    """
    def __init__(self, job, config_file, coupling_subdomain,
                 write_object, function_space=None):
        self._job = job
        self._config_file = config_file
        self._coupling_boundary = self._job.domain.get_subdomain(coupling_subdomain)
        self._coupling_subdomain = coupling_subdomain # a string
        self._write_object = write_object
        if function_space is None:
            self._function_space = self._job.solver.V
        else:
            self._function_space = function_space
        self._dt = None
        self._coupling_expression = None
        self._update_boundary_func = None
        self._coupling_data_func = None
        self._adapter = None
        self._instantiated = False
        self._dirichlet_normal_direction = None

    def instantiate_adapter(self):
        """
        instantiate the adaptor
        """
        self._instantiated = True
        return fenicsprecice.Adapter(adapter_config_filename=self._config_file)

    @property
    def coupling_boundary(self):
        return self._coupling_boundary

    @coupling_boundary.setter
    def coupling_boundary(self, subdomain):
        if isinstance(subdomain, FEN.SubDomain) or isinstance(subdomain, FenicsSubDomain):
            self._coupling_boundary = subdomain
        else:
            raise TypeError(f"expected fenics.SubDomain or FenicsSubDomain, but received {type(subdomain)}")

    @property
    def write_object(self):
        return self._write_object

    @write_object.setter
    def write_object(self, write_obj):
        if isinstance(write_obj, dolfin.function.function.Function):
            self._write_object = write_obj
        else:
            raise TypeError("expected fenics.Expression,"
                            f" but received {type(write_obj)}")

    @property
    def function_space(self):
        return self._function_space

    @function_space.setter
    def function_space(self, _function):
        if isinstance(_function, dolfin.function.function.Function):
            self._function_space = _function
        else:
            raise TypeError("expected fenics.function.function.Function,"
                            f" but received {type(_function)}")

    @property
    def update_boundary_func(self):
        return self._update_boundary_func

    @update_boundary_func.setter
    def update_boundary_func(self, update_func):
        """
        here a callable function is expected which in principle adds
        a boundary condition to the list of boundary conditions
        The function requires three arguments:
            - job
            - coupling_expression
            - coupling_subdomain
        An example of such functions:
        for a dirichlet boundary:
        job.adapter_conf.update_boundary_func = some_func
        where,
        def some_func(coupling_expression, coupling_subdomain, job):
            job.domain.boundary(bc_type='dirichlet', subdomain_name=coupling_subdomain, expression=coupling_expression)

        """
        if callable(update_func):
            self._update_boundary_func = update_func
        else:
            raise TypeError(f'expected a function but received a {type(update_func)}')

    def update_coupling_boundary(self, _adapter):
        self._coupling_expression = _adapter.create_coupling_expression()
        self.update_boundary_func( self._coupling_expression, self._coupling_subdomain, job=self._job)

    @property
    def coupling_expression(self):
        if self._coupling_expression is not None:
            return self._coupling_expression
        else:
            raise NotSetCorrectlyError("The adapter_conf is not set correctly!"
                                       "No coupling_expression was instantiated in first place!")

    @property
    def coupling_data(self):
        if self._coupling_data_func is None:
            raise NotSetCorrectlyError("The adaptor configuration not correctly set!\n"
                                       "No coupling_data_function is set!")
        return self._coupling_data_func(self, self._job)

    @property
    def coupling_data_func(self):
        return self._coupling_data_func

    @coupling_data_func.setter
    def coupling_data_func(self, func):
        """
        here a callable function is expected which in principle returns the coupling data
        later this is passed to job.adapter.write_data()
        An example of such functions:
        for a dirichlet boundary:
        job.adapter_conf.coupling_data_func = some_func
        The function requires two arguments:
                - config: this represents job.adapter_conf
                - job: this represents the job itself
        An example of such a function:
            def some_user_defined_func(config, job):
                return job.solution
        """

        if callable(func):
            self._coupling_data_func = func
        else:
            raise ValueError("the given value must be a callable function")

    def predefined_coupling(self, coupling_type, flux_direction=None):
        """
        job.adapter_conf allows you to define functions which updates couplings during the job.run().
        There are two presets: 'Dirichlet and Neumann"
        if you want to use these two presets: you just need to:
        job.adapter_config.predefined_coupling(coupling_type, flux_direction=None)
        Args:
            coupling_type(str): can be "Dirichlet" or "Neumann"
            flux_direction(str): if you use "Dirichlet", you need to pass
                                in the direction of flux, 'x', 'y', or 'z'
        """
        if coupling_type not in ["Dirichlet", "dirichlet", "Neumann", "neumann"]:
            raise ValueError("coupling_type must be of"
                             '"Dirichlet", "dirichlet", "Neumann", "neumann"')
        elif coupling_type in ["Dirichlet", "dirichlet"]:
            try:
                _dict = {'x': 0, 'y': 1, 'z': 2}
                self._dirichlet_normal_direction = _dict[flux_direction]
            except Exception as err_msg:
                raise Exception(err_msg)

            self.update_boundary_func = self.dirichlet_update_coupling
            self.coupling_data_func = self.dirichlet_coupling_data

        elif coupling_type in ["Neumann", "neumann"]:
            self.update_boundary_func = self.neumann_update_coupling
            self.coupling_data_func = self.neumann_coupling_data

    @staticmethod
    def dirichlet_update_coupling(coupling_expression, coupling_subdomain, job):
        job.domain.boundary(bc_type='dirichlet', subdomain_name=coupling_subdomain, expression=coupling_expression)

    @staticmethod
    def dirichlet_coupling_data(config, job):
        job.solver.flux.rename("Flux", "")
        job.solver.calc_flux()
        return FEN.interpolate(job.solver.flux.sub(0), config.write_object.function_space())

    @staticmethod
    def neumann_update_coupling(coupling_expression, coupling_subdomain, job):
        if job.solver.F is None:
            print(f"job.solver.F is None!")
        elif coupling_expression is None:
            print("coupling_expression is None")

        job.solver.F = job.solver.F + job.solver.v * coupling_expression * FEN.ds

    @staticmethod
    def neumann_coupling_data(config, job):
        return job.solver.solution

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
        - rhs (left hand side of the equation, normally found by calling FEN.lhs(F)
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

    def __init__(self, job):
        self._job = job
        self._V = FEN.FunctionSpace(job.mesh, job.input.element_type,
                                    job.input.element_order)  # finite element volume space
        if job.input.element_order > 1:
            self._V_g = FEN.VectorFunctionSpace(job.mesh, job.input.element_type, job.input.element_order - 1)
        else:
            self._V_g = FEN.VectorFunctionSpace(job.mesh, job.input.element_type, job.input.element_order)

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
        self._solution = FEN.Function(self._V)
        self._flux = FEN.Function(self._V_g)
        self.time_dependent_expressions = []
        self.assigned_u = None

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
        if isinstance(expression, FEN.Expression) \
                or isinstance(expression, FEN.Constant):
            self._f = expression
        else:
            raise ValueError("job.solver.f must be of type fenics.Expression or fenics.Constant")

    # job_d.function_space.set_parameter(key='f', expression='beta - 2 - 2*alpha', degree=2, alpha=alpha, beta=beta)
    def set_expression(self, key, expression, **kwargs):

        if key == 'f':
            self.f = FEN.Expression(expression, **kwargs)
        else:
            self._extra_parameters[key] = [expression, kwargs]

    def update_F(self, value):
        try:
            self._F = self._F + value
        except Exception as err_msg:
            raise Exception(err_msg)

    @property
    def F(self):
        return self._F

    @F.setter
    def F(self, equation):
        if isinstance(equation, str):
            self._string_equation = equation
        else:
            self._F = equation

    def _evaluate_equation(self):
        if self._F is None:
            if self.assigned_u is None and not self._u_n is None:
                self.assigned_u = self._u_n
            elif self._job.input.n_steps > 1 and self.assigned_u is None:
                raise ValueError(
                    "When the timesteps are larger that one, job.solver.assigned_u or self.solver.u_n should be specified")
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

            u = self._u
            v = self._v
            u_n = self._u_n
            dot = FEN.dot
            grad = FEN.grad
            dx = FEN.dx
            dt = self._dt
            inner = FEN.inner
            if not self._f is None:
                f = self._f

            try:
                self._F = eval(self._string_equation)
                self._lhs = FEN.lhs(self._F)
                self._rhs = FEN.rhs(self._F)
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

    def set_initial_condition(self, function=None, expression=None, interpolate=True, project=False,
                              function_space=None, **kwargs):
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
                FEN.Expression(expression, **kwargs)
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
        self._lhs = FEN.lhs(self.F)
        self._rhs = FEN.rhs(self.F)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

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

class NotSetCorrectlyError(Exception):
    "raised when the object is not configured correctly!"
    pass