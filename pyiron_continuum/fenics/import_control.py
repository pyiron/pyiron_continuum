# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

"""
A centralized location to control the import alarm for all the (optional!) fenics imports.
"""

from pyiron_base import ImportAlarm

__author__ = "Liam Huber"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH - "
    "Computational Materials Design (CM) Department"
)
__version__ = "0.1"
__maintainer__ = "Liam Huber"
__email__ = "huber@mpie.de"
__status__ = "development"
__date__ = "Oct 4, 2021"

try:
    import fenics as FEN
    import mshr

    from dolfin.common.plotting import (
        _has_matplotlib,
        _all_plottable_types,
        _plot_x3dom,
        _matplotlib_plottable_types,
        mplot_mesh,
        mplot_dirichletbc,
        mplot_expression,
        mplot_function,
        mplot_meshfunction,
        _meshfunction_types
    )
    import dolfin.common.plotting as dcp
    import dolfin.cpp as cpp
    import ufl
    import sympy

    fenics_alarm = ImportAlarm()
except ImportAlarm:
    fenics_alarm = ImportAlarm(
        'Fenics functionality requires the `fenics` and `mshr` modules (and their dependencies) specified as extra '
        'requirements. Please install these and try again.'
    )


class FenicsModules:
    @fenics_alarm
    def __init__(self):
        pass

    @property
    def FEN(self):
        return FEN

    @property
    def mshr(self):
        return mshr

    @property
    def sympy(self):
        return sympy

    @property
    def ufl(self):
        return ufl

    @property
    def cpp(self):
        return cpp

    @property
    def dcp(self):
        return dcp


fm = FenicsModules()
