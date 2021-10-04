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

with ImportAlarm(
        'Fenics functionality requires the `fenics` and `mshr` modules (and their dependencies) specified as extra '
        'requirements. Please install these and try again.'
) as fenics_alarm:
    import fenics as FEN
    import mshr
    import dolfin.common.plotting as dcp
    import dolfin.cpp as cpp
    import ufl
    import sympy


class FenicsModules:
    @fenics_alarm
    def __init__(self):
        pass

    @property
    @fenics_alarm
    def FEN(self):
        return FEN

    @property
    @fenics_alarm
    def mshr(self):
        return mshr

    @property
    @fenics_alarm
    def sympy(self):
        return sympy

    @property
    @fenics_alarm
    def ufl(self):
        return ufl

    @property
    @fenics_alarm
    def cpp(self):
        return cpp

    @property
    @fenics_alarm
    def dcp(self):
        return dcp


fm = FenicsModules()
