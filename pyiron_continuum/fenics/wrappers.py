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
from typing import Type


class StringInputParser(ABC):
    """
    Not fool proof, but does its best to ensure that a string is valid and will execute in the scope of `known_elements`
    (provided by the developer) and `**kwargs` (provided by the user).
    """
    def __init__(self, input_string: str, __verbose=True, **kwargs):
        self._splitting_elements = [
            r"\*?\*", r"\+", r"-", r"/",
            r"\(", r"\)", r",", r"\[", r"\]",
            r"<", r">", r"=", r"==",
            r"and", r"or", r"is", r"not"
        ]
        self._verbose = __verbose
        self._test_kwargs(kwargs)
        self._test_elements(input_string, kwargs)

    @property
    @abstractmethod
    def _known_elements(self):
        """Classes, methods, or strings which should be allowed to appear in the expression"""
        pass

    @property
    def known_elements(self):
        return [k.__name__ if hasattr(k, '__name__') else k for k in self._known_elements]

    def _test_kwargs(self, kwargs):
        failures = {}
        for key, value in kwargs.items():
            try:
                float(value)
            except:
                try:  # Recursively check if the value is an allowable type for this parser
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
        if failures and self._verbose:
            raise ValueError(f"Got an unexpected symbol(s) '{failures}'")


class FenicsWrapper(HasStorage, ABC):
    """
    A master class for behavior like
    >>> my_fenics_obj = FenicsWrapperChild('some fenics code', k='v')
    >>> my_fenics_obj.set('I changed my mind, here is different fenics-compatible code', k='w', l=10)
    >>> my_fenics_obj()  # Returns a instance of a fenics object
    or
    >>> my_fenics_obj(some_important_arg)  # in case generating the fenics object requires more data
    Note that in the latter case, the extra data had better always be available (e.g. the function space in the case of
    meshes), or we're not serializing enough data!!!

    All args and kwargs get automatically stored, but the developer must provide an appropriate parser (the `_parser`
    property) and a method for turning those snippets into a fenics object (the `_generate` method).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._as_fenics = None
        self.set(*args, **kwargs)

    @property
    @abstractmethod
    def _parser(self) -> Type[StringInputParser]:
        pass

    def _parse(self, input_string, **kwargs):
        return self._parser(input_string, **kwargs)

    @abstractmethod
    def _generate(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        self._as_fenics = self._generate(*args, **kwargs)

    def set(self, *args, **kwargs):
        for arg in args:
            # A little inefficient, as we parse the kwargs multiple times...
            self._parse(arg, **kwargs)
        self.storage.args = args
        self.storage.kwargs = kwargs
        self._as_fenics = None

    def __call__(self, *args, **kwargs):
        if self._as_fenics is None or len(args) > 0 or len(kwargs) > 0:
            self.generate(*args, **kwargs)
        return self._as_fenics


class BCParser(StringInputParser):
    @property
    def _known_elements(self):
        return ["x", near, Constant, Expression]


class DirichletBC(FenicsWrapper):
    @property
    def _parser(self):
        return BCParser

    def _generate(self, function_space):
        value, condition = self.storage.args

        def boundary_func(x, on_boundary):
            try:
                return on_boundary and eval(condition)
            except Exception as err_msg:
                print(err_msg)

        return FEN.DirichletBC(function_space, eval(value), boundary_func)


class MeshParser(StringInputParser):
    @property
    def _known_elements(self):
        return [
            Point,
            BoxMesh, RectangleMesh, UnitSquareMesh, UnitCubeMesh, UnitDiscMesh, UnitTriangleMesh, UnitIntervalMesh,
            generate_mesh, Tetrahedron, Box, Rectangle, Circle
        ]


class Mesh(FenicsWrapper):
    @property
    def _parser(self):
        return MeshParser

    def _generate(self):
        for k, v in self.storage.kwargs.items():
            exec(f'{k} = {v}')
        mesh = eval(self.storage.args[0])
        if not isinstance(mesh, FenicsMesh.Mesh):
            raise TypeError(f'Expected the mesh to be of type dolfin.cpp.mesh.Mesh, but got {type(mesh)}. You may '
                            f'need to wrap your expression in "generate_mesh()" to convert a mshr mesh to a dolphin '
                            f'mesh.')
        return mesh


