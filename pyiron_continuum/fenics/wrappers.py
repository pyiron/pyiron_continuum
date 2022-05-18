# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
Classes for matching fenics objects to serializable pyiron objects.
"""

import re
from abc import ABC, abstractmethod
import fenics as FEN
from fenics import (
    near, Constant, Expression,
    Point, BoxMesh, RectangleMesh, UnitSquareMesh, UnitCubeMesh, UnitDiscMesh, UnitTriangleMesh, UnitIntervalMesh
)
from mshr import (
    generate_mesh, Tetrahedron, Box, Rectangle, Circle
)
from dolfin.cpp import mesh as FenicsMesh
from pyiron_base import HasStorage


class StringInputParser:
    """
    Not fool proof, but does its best to ensure that a string is valid and will execute in the scope of `known_elements`
    (provided by the developer at instantiation) and `**kwargs` (provided by the user at call).
    """
    def __init__(self, known_elements):
        self._splitting_elements = [
            r"\*?\*", r"\+", r"-", r"/",
            r"\(", r"\)", r",", r"\[", r"\]",
            r"<", r">", r"=", r"==",
            r"and", r"or", r"is", r"not",
            r"'", r'"'
        ]
        self._known_elements = known_elements

    def __call__(self, input_string: str, __verbose=True, **kwargs):
        self._test_kwargs(kwargs, __verbose)
        self._test_elements(input_string, kwargs, __verbose)

    @property
    def known_elements(self):
        return [k.__name__ if hasattr(k, '__name__') else k for k in self._known_elements]

    def _test_kwargs(self, kwargs, __verbose):
        failures = {}
        for key, value in kwargs.items():
            try:
                float(value)
            except:
                try:  # Recursively check if the value is an allowable type for this parser
                    self(value, __verbose=False)
                except:
                    failures[key] = value
        if failures and __verbose:
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

    def _test_elements(self, input_string, kwargs, __verbose):
        failures = []
        for e in self._split_input(input_string):
            if e in self.known_elements:
                continue
            elif e.isnumeric():
                continue
            elif e in kwargs.keys():
                continue
            else:
                try:
                    float(e)
                    continue
                except:
                    failures.append(e)
        if failures and __verbose:
            raise ValueError(f"Got an unexpected symbol(s) '{failures}'")


class FenicsWrapper(HasStorage, ABC):
    """
    A master class for behavior like
    >>> my_fenics_obj = FenicsWrapperChild('some fenics-compatible code', k='v')
    >>> my_fenics_obj.set('I changed my mind, here is different fenics-compatible code', k='w', l=10)
    >>> my_fenics_obj()  # Returns a instance of a fenics object
    or
    >>> my_fenics_obj(some_important_arg)  # in case generating the fenics object requires more data
    Note that in the latter case, the extra data had better always be available (e.g. the function space in the case of
    meshes), or we're not serializing enough data!!!

    The input string and kwargs get automatically stored (as long as the kwargs contain simple, serializable stuff like
    floats), but the developer must provide an appropriate parser (the `_parser` property) and a method for turning
    the input into a fenics object (the `_generate` method).
    """

    def __init__(self, input_string, **kwargs):
        super().__init__()
        self._as_fenics = None
        self.set(input_string, **kwargs)

    @property
    @abstractmethod
    def _parser(self) -> StringInputParser:
        pass

    def _parse(self, input_string, **kwargs):
        return self._parser(input_string, **kwargs)

    @abstractmethod
    def _generate(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        self._as_fenics = self._generate(*args, **kwargs)

    def set(self, input_string, **kwargs):
        self._parse(input_string, **kwargs)
        self.storage.input_string = input_string
        self.storage.kwargs = kwargs
        self._as_fenics = None

    def __call__(self, *args, **kwargs):
        if self._as_fenics is None or len(args) > 0 or len(kwargs) > 0:
            self.generate(*args, **kwargs)
        return self._as_fenics

    @property
    def input_string(self):
        return self.storage.input_string

    @property
    def kwargs(self):
        return self.storage.kwargs

    def __str__(self):
        return self.input_string


class Condition(FenicsWrapper):
    """
    Creates a function that tests the spatial value (canonically `x` in fenics) and whether that point is `on_boundary`.
    The input string defines the condition to be evaluated, and all kwargs are exec'd beforehand.
    """
    @property
    def _parser(self):
        return StringInputParser(known_elements=["x", near])

    def _generate(self):
        for k, v in self.kwargs.items():
            exec(f'{k} = {v}')

        def boundary_func(x, on_boundary):
            try:
                return on_boundary and eval(self.input_string)
            except Exception as err_msg:
                print(err_msg)

        return boundary_func


class Value(FenicsWrapper):
    """
    Creates a fenics `Constant` or `Expression` depending whether `degree` is included in the kwargs.

    The fenics class ("Constant" or "Expression") can be omitted from the input string, and all kwargs are passed
    directly to the fenics class itself.
    """
    @property
    def _parser(self) -> StringInputParser:
        return StringInputParser(known_elements=["x", "exp", "pow"])

    def _generate(self):
        if 'degree' in self.kwargs.keys():
            return self._generate_expression(self.input_string, **self.kwargs)
        else:
            return self._generate_constant(self.input_string, **self.kwargs)

    @staticmethod
    def _generate_expression(value_tuple, **kwargs):
        return Expression(eval(value_tuple), **kwargs)

    @staticmethod
    def _generate_constant(value_tuple, **kwargs):
        return Constant(eval(value_tuple), **kwargs)


class DirichletBC(HasStorage):
    def __init__(self, value: Value, condition: Condition):
        super().__init__()
        self.storage.value = value
        self.storage.condition = condition

    @property
    def value(self):
        return self.storage.value

    @property
    def condition(self):
        return self.storage.condition

    def __call__(self, function_space):
        return FEN.DirichletBC(function_space, self.value(), self.condition())

    def __str__(self):
        return str(self.value) + ", " + str(self.condition)


class Mesh(FenicsWrapper):
    """
    Creates a spatial mesh.

    The input string should include all fenics code to be evaluated, including the class to be called, while all kwargs
    are exec'd beforehand.
    """
    @property
    def _parser(self):
        return StringInputParser(known_elements=[
            Point,
            BoxMesh, RectangleMesh, UnitSquareMesh, UnitCubeMesh, UnitDiscMesh, UnitTriangleMesh, UnitIntervalMesh,
            generate_mesh, Tetrahedron, Box, Rectangle, Circle
        ])

    def _generate(self):
        for k, v in self.kwargs.items():
            exec(f'{k} = {v}')
        mesh = eval(self.input_string)
        if not isinstance(mesh, FenicsMesh.Mesh):
            raise TypeError(f'Expected the mesh to be of type dolfin.cpp.mesh.Mesh, but got {type(mesh)}. You may '
                            f'need to wrap your input string in "generate_mesh()" to convert a mshr mesh to a dolphin '
                            f'mesh.')
        return mesh
