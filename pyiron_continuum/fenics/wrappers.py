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
from pyiron_base import PyironFactory, HasStorage


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