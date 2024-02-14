# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Factories for Fenics-related object creation.
"""

from pyiron_base import ImportAlarm, PyironFactory, HasStorage
from pyiron_continuum.fenics.wrappers import DirichletBC, Value, Condition

with ImportAlarm(
    "fenics functionality requires the `fenics`, `mshr` modules (and their dependencies) specified as extra"
    "requirements. Please install it and try again."
) as fenics_alarm:
    import fenics as FEN

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
        if self._job.mesh is None:
            raise NotSetCorrectlyError(
                "Before accessing job.solver, job.mesh should be defined"
                "You can use job.input.mesh to set job.mesh"
            )
        else:
            self._V = func_space_class(
                job.mesh, job.input.element_type, job.input.element_order
            )  # finite element volume space
            if job.input.element_order > 1:
                self._V_g = FEN.VectorFunctionSpace(
                    job.mesh, job.input.element_type, job.input.element_order - 1
                )
            else:
                self._V_g = FEN.VectorFunctionSpace(
                    job.mesh, job.input.element_type, job.input.element_order
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


class BoundaryConditions(PyironFactory, HasStorage):
    def __init__(self):
        PyironFactory.__init__(self)
        HasStorage.__init__(self)

    def append(self, value_string, condition_string, condition_kwargs=None, **value_kwargs):
        expression = Value(value_string, **value_kwargs)
        condition_kwargs = condition_kwargs if condition_kwargs is not None else {}
        condition = Condition(condition_string, **condition_kwargs)
        self.storage.append(DirichletBC(expression, condition))

    def list(self):
        return [str(v) for v in self.storage.values()]

    def clear(self):
        self.storage.clear()

    def pop(self, i):
        return self.storage.pop(i)

    def __call__(self, function_space):
        return [bc(function_space) for bc in self.storage.values()]
